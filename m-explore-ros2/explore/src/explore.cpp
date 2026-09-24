/*********************************************************************
 *
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2008, Robert Bosch LLC.
 *  Copyright (c) 2015-2016, Jiri Horner.
 *  Copyright (c) 2021, Carlos Alvarez, Juan Galvis.
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of the Jiri Horner nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 *********************************************************************/

#include <explore/explore.h>

#include <algorithm>
#include <mutex>
#include <thread>

#include "nav2_costmap_2d/cost_values.hpp"

namespace explore
{
Explore::Explore()
  : Node("explore_node")
  , logger_(this->get_logger())
  , tf_buffer_(this->get_clock())
  , tf_listener_(tf_buffer_)
  , costmap_client_(*this, &tf_buffer_)
  , prev_distance_(0)
  , last_markers_count_(0)
{
  double timeout;
  double min_frontier_size;
  this->declare_parameter<float>("planner_frequency", 1.0);
  this->declare_parameter<float>("progress_timeout", 30.0);
  this->declare_parameter<bool>("visualize", false);
  this->declare_parameter<float>("potential_scale", 1e-3);
  this->declare_parameter<float>("orientation_scale", 0.0);
  this->declare_parameter<float>("gain_scale", 1.0);
  this->declare_parameter<float>("min_frontier_size", 0.5);
  this->declare_parameter<bool>("return_to_init", false);
  this->declare_parameter<float>("blacklist_radius", 1.0);
  this->declare_parameter<float>("min_goal_distance", 1.5);
  this->declare_parameter<float>("goal_clearance", 0.7);
  this->declare_parameter<float>("max_goal_backoff", 2.0);
  this->declare_parameter<int>("max_no_goal_cycles", 3);

  this->get_parameter("planner_frequency", planner_frequency_);
  this->get_parameter("progress_timeout", timeout);
  this->get_parameter("visualize", visualize_);
  this->get_parameter("potential_scale", potential_scale_);
  this->get_parameter("orientation_scale", orientation_scale_);
  this->get_parameter("gain_scale", gain_scale_);
  this->get_parameter("min_frontier_size", min_frontier_size);
  this->get_parameter("return_to_init", return_to_init_);
  this->get_parameter("robot_base_frame", robot_base_frame_);
  this->get_parameter("blacklist_radius", blacklist_radius_);
  this->get_parameter("min_goal_distance", min_goal_distance_);
  this->get_parameter("goal_clearance", goal_clearance_);
  this->get_parameter("max_goal_backoff", max_goal_backoff_);
  this->get_parameter("max_no_goal_cycles", max_no_goal_cycles_);

  progress_timeout_ = timeout;
  move_base_client_ =
      rclcpp_action::create_client<nav2_msgs::action::NavigateToPose>(
          this, ACTION_NAME);

  search_ = frontier_exploration::FrontierSearch(costmap_client_.getCostmap(),
                                                 potential_scale_, gain_scale_,
                                                 min_frontier_size, logger_);

  if (visualize_) {
    marker_array_publisher_ =
        this->create_publisher<visualization_msgs::msg::MarkerArray>("explore/"
                                                                     "frontier"
                                                                     "s",
                                                                     10);
  }

  // Publisher for exploration status
  rclcpp::QoS status_qos(10);
  status_qos.transient_local();
  status_pub_ = this->create_publisher<explore_lite_msgs::msg::ExploreStatus>("explore/status", status_qos);

  // Subscription to resume or stop exploration
  resume_subscription_ = this->create_subscription<std_msgs::msg::Bool>(
      "explore/resume", 10,
      std::bind(&Explore::resumeCallback, this, std::placeholders::_1));

  RCLCPP_INFO(logger_, "Waiting to connect to move_base nav2 server");
  move_base_client_->wait_for_action_server();
  RCLCPP_INFO(logger_, "Connected to move_base nav2 server");

  if (return_to_init_) {
    RCLCPP_INFO(logger_, "Getting initial pose of the robot");
    geometry_msgs::msg::TransformStamped transformStamped;
    std::string map_frame = costmap_client_.getGlobalFrameID();
    try {
      transformStamped = tf_buffer_.lookupTransform(
          map_frame, robot_base_frame_, tf2::TimePointZero);
      initial_pose_.position.x = transformStamped.transform.translation.x;
      initial_pose_.position.y = transformStamped.transform.translation.y;
      initial_pose_.orientation = transformStamped.transform.rotation;
    } catch (tf2::TransformException& ex) {
      RCLCPP_ERROR(logger_, "Couldn't find transform from %s to %s: %s",
                   map_frame.c_str(), robot_base_frame_.c_str(), ex.what());
      return_to_init_ = false;
    }
  }

  exploring_timer_ = this->create_wall_timer(
      std::chrono::milliseconds((uint16_t)(1000.0 / planner_frequency_)),
      [this]() { makePlan(); });
  // Start exploration right away
  auto status_msg = explore_lite_msgs::msg::ExploreStatus();
  status_msg.status = explore_lite_msgs::msg::ExploreStatus::EXPLORATION_STARTED;
  status_pub_->publish(status_msg);
  makePlan();
}

Explore::~Explore()
{
  stop();
}

void Explore::resumeCallback(const std_msgs::msg::Bool::SharedPtr msg)
{
  if (msg->data) {
    resume();
  } else {
    stop();
  }
}

void Explore::visualizeFrontiers(
    const std::vector<frontier_exploration::Frontier>& frontiers)
{
  const auto blue = std_msgs::msg::ColorRGBA().set__b(1.0).set__a(0.5);
  const auto red = std_msgs::msg::ColorRGBA().set__r(1.0).set__a(0.5);
  const auto green = std_msgs::msg::ColorRGBA().set__g(1.0).set__a(0.5);

  RCLCPP_DEBUG(logger_, "visualising %lu frontiers", frontiers.size());
  visualization_msgs::msg::MarkerArray markers_msg;
  std::vector<visualization_msgs::msg::Marker>& markers = markers_msg.markers;
  visualization_msgs::msg::Marker m;

  m.header.frame_id = costmap_client_.getGlobalFrameID();
  m.header.stamp = this->now();
  m.ns = "frontiers";
  m.scale.x = 1.0;
  m.scale.y = 1.0;
  m.scale.z = 1.0;
  m.color.r = 0;
  m.color.g = 0;
  m.color.b = 255;
  m.color.a = 255;
  // m.lifetime defaults to 0, means lives forever
  m.frame_locked = true;

  // weighted frontiers are always sorted
  double min_cost = frontiers.empty() ? 0. : frontiers.front().cost;

  m.action = visualization_msgs::msg::Marker::ADD;
  size_t id = 0;
  for (auto& frontier : frontiers) {
    m.type = visualization_msgs::msg::Marker::POINTS;
    m.id = int(id);
    m.pose.position.x = 0.0;
    m.pose.position.y = 0.0;
    m.pose.position.z = 0.0;
    m.scale.x = 0.1;
    m.scale.y = 0.1;
    m.scale.z = 0.1;
    m.points = frontier.points;
    if (goalOnBlacklist(frontier.centroid)) {
      m.color = red;
    } else {
      m.color = blue;
    }
    markers.push_back(m);
    ++id;
    m.type = visualization_msgs::msg::Marker::SPHERE;
    m.id = int(id);
    m.pose.position = frontier.centroid;
    // scale frontier according to its cost (costier frontiers will be smaller)
    double scale = std::min(std::abs(min_cost * 0.4 / frontier.cost), 0.5);
    m.scale.x = scale;
    m.scale.y = scale;
    m.scale.z = scale;
    m.points = {};
    m.color = green;
    markers.push_back(m);
    ++id;
  }
  size_t current_markers_count = markers.size();

  // delete previous markers, which are now unused
  m.action = visualization_msgs::msg::Marker::DELETE;
  for (; id < last_markers_count_; ++id) {
    m.id = int(id);
    markers.push_back(m);
  }

  last_markers_count_ = current_markers_count;
  marker_array_publisher_->publish(markers_msg);
}

void Explore::makePlan()
{
  // Robot pose. If TF is not available, skip this cycle instead of searching
  // from (0,0) (the old behaviour, which ranked frontiers from the map origin).
  geometry_msgs::msg::Pose pose;
  if (!costmap_client_.getRobotPose(pose)) {
    return;
  }

  // get frontiers sorted according to cost
  auto frontiers = search_.searchFrom(pose.position);
  RCLCPP_DEBUG(logger_, "found %lu frontiers", frontiers.size());
  for (size_t i = 0; i < frontiers.size(); ++i) {
    RCLCPP_DEBUG(logger_, "frontier %zd cost: %f", i, frontiers[i].cost);
  }

  if (frontiers.empty()) {
    finishExploration("no frontiers left");
    return;
  }

  // publish frontiers as visualization markers
  if (visualize_) {
    visualizeFrontiers(frontiers);
  }

  // 1) Commit to the goal we are already driving to.
  // Re-picking the "best" frontier every cycle made the goal jump to a new
  // direction whenever the robot moved a few cm (which cell of a large
  // frontier is closest changes all the time). With a rotate-in-place
  // controller that means the robot only turns and never drives. So keep the
  // current goal until it is reached, aborted, timed out, or the unknown area
  // it was chosen for has been observed.
  if (goal_active_ && has_goal_) {
    if (goalStillUseful(frontiers)) {
      const double dist_to_goal =
          std::hypot(pose.position.x - current_goal_.x,
                     pose.position.y - current_goal_.y);
      // progress = robot actually got >= 10 cm closer
      if (dist_to_goal < prev_distance_ - 0.1) {
        last_progress_ = this->now();
        prev_distance_ = dist_to_goal;
      }
      if ((this->now() - last_progress_ >
           tf2::durationFromSec(progress_timeout_)) &&
          !resuming_) {
        RCLCPP_WARN(logger_,
                    "No progress towards (%.2f, %.2f) for %.0f s, "
                    "blacklisting it",
                    current_goal_.x, current_goal_.y, progress_timeout_);
        frontier_blacklist_.push_back(current_goal_);
        frontier_blacklist_.push_back(current_centroid_);
        has_goal_ = false;
        // fall through and choose a new goal
      } else {
        resuming_ = false;
        return;  // keep driving to the committed goal
      }
    } else {
      RCLCPP_INFO(logger_,
                  "Area around goal (%.2f, %.2f) already explored, "
                  "choosing a new goal",
                  current_goal_.x, current_goal_.y);
      has_goal_ = false;
    }
  }
  resuming_ = false;

  // 2) Choose a new goal: best (lowest cost) frontier that is not blacklisted
  // AND has a safe goal point that Nav2 can reach (not the raw centroid).
  geometry_msgs::msg::Point target_position;
  geometry_msgs::msg::Point target_centroid;
  bool found = false;
  bool skipped_unsafe = false;
  for (const auto& f : frontiers) {
    if (goalOnBlacklist(f.centroid)) {
      continue;
    }
    geometry_msgs::msg::Point candidate;
    if (!selectGoal(f, pose.position, candidate)) {
      RCLCPP_DEBUG(logger_,
                   "frontier at (%.2f, %.2f) has no obstacle-free goal point",
                   f.centroid.x, f.centroid.y);
      skipped_unsafe = true;
      continue;
    }
    if (goalOnBlacklist(candidate)) {
      continue;
    }
    target_position = candidate;
    target_centroid = f.centroid;
    found = true;
    break;
  }

  if (!found) {
    // A goal we just dropped (timed out / explored) may still be active in
    // Nav2: cancel it so the robot does not keep driving there.
    if (goal_active_) {
      move_base_client_->async_cancel_all_goals();
      goal_active_ = false;
    }
    // Every frontier is blacklisted -> done. If some were only skipped because
    // no safe goal exists right now, wait a few cycles (the map may still
    // change) before declaring completion.
    if (!skipped_unsafe || ++no_goal_cycles_ >= max_no_goal_cycles_) {
      finishExploration(
          "all remaining frontiers are blacklisted or unreachable");
    } else {
      RCLCPP_INFO(logger_,
                  "No reachable frontier this cycle (%d/%d), retrying",
                  no_goal_cycles_, max_no_goal_cycles_);
    }
    return;
  }
  no_goal_cycles_ = 0;

  const double dist_to_goal = std::hypot(pose.position.x - target_position.x,
                                         pose.position.y - target_position.y);
  last_progress_ = this->now();
  prev_distance_ = dist_to_goal;

  current_goal_ = target_position;
  current_centroid_ = target_centroid;
  has_goal_ = true;

  RCLCPP_INFO(logger_, "New exploration goal (%.2f, %.2f), %.1f m away",
              target_position.x, target_position.y, dist_to_goal);

  if (visualize_) {
    visualization_msgs::msg::MarkerArray goal_markers;
    visualization_msgs::msg::Marker g;
    g.header.frame_id = costmap_client_.getGlobalFrameID();
    g.header.stamp = this->now();
    g.ns = "explore_goal";
    g.id = 0;
    g.type = visualization_msgs::msg::Marker::SPHERE;
    g.action = visualization_msgs::msg::Marker::ADD;
    g.pose.position = target_position;
    g.pose.orientation.w = 1.0;
    g.scale.x = 0.4;
    g.scale.y = 0.4;
    g.scale.z = 0.4;
    g.color.r = 1.0;
    g.color.g = 1.0;
    g.color.a = 1.0;  // yellow = goal actually sent to Nav2
    goal_markers.markers.push_back(g);
    marker_array_publisher_->publish(goal_markers);
  }

  RCLCPP_DEBUG(logger_, "Sending goal to move base nav2");

  // send goal to move_base if we have something new to pursue
  auto goal = nav2_msgs::action::NavigateToPose::Goal();
  goal.pose.pose.position = target_position;
  // Face the direction of travel so the robot does not have to turn on the
  // spot next to an obstacle when it arrives (old code: always yaw = 0).
  const double yaw = std::atan2(target_position.y - pose.position.y,
                                target_position.x - pose.position.x);
  goal.pose.pose.orientation.z = std::sin(yaw / 2.0);
  goal.pose.pose.orientation.w = std::cos(yaw / 2.0);
  goal.pose.header.frame_id = costmap_client_.getGlobalFrameID();
  goal.pose.header.stamp = this->now();

  goal_active_ = true;
  auto send_goal_options = rclcpp_action::Client<
      nav2_msgs::action::NavigateToPose>::SendGoalOptions();

  send_goal_options.goal_response_callback =
      [this](const NavigationGoalHandle::SharedPtr& goal_handle) {
        if (!goal_handle) {
          RCLCPP_ERROR(logger_, "Goal was REJECTED by the action server");
          goal_active_ = false;
        } else {
          active_goal_id_ = goal_handle->get_goal_id();
          RCLCPP_DEBUG(logger_, "Goal ACCEPTED, uuid: %s",
            rclcpp_action::to_string(active_goal_id_).c_str());
        }
      };

  send_goal_options.result_callback =
      [this, target_position,
       target_centroid](const NavigationGoalHandle::WrappedResult& result) {
        reachedGoal(result, target_position, target_centroid);
      };
  move_base_client_->async_send_goal(goal, send_goal_options);
}

bool Explore::hasClearance(double wx, double wy, double clearance)
{
  nav2_costmap_2d::Costmap2D* costmap = costmap_client_.getCostmap();
  unsigned int mx, my;
  if (!costmap->worldToMap(wx, wy, mx, my)) {
    return false;
  }
  // the goal cell itself must be known free space
  if (costmap->getCost(mx, my) != nav2_costmap_2d::FREE_SPACE) {
    return false;
  }
  const int r =
      static_cast<int>(std::ceil(clearance / costmap->getResolution()));
  const int size_x = static_cast<int>(costmap->getSizeInCellsX());
  const int size_y = static_cast<int>(costmap->getSizeInCellsY());
  for (int dy = -r; dy <= r; ++dy) {
    for (int dx = -r; dx <= r; ++dx) {
      if (dx * dx + dy * dy > r * r) {
        continue;
      }
      const int cx = static_cast<int>(mx) + dx;
      const int cy = static_cast<int>(my) + dy;
      if (cx < 0 || cy < 0 || cx >= size_x || cy >= size_y) {
        continue;
      }
      const unsigned char c = costmap->getCost(static_cast<unsigned int>(cx),
                                               static_cast<unsigned int>(cy));
      // unknown cells are fine (that is the frontier), occupied cells are not
      if (c != nav2_costmap_2d::NO_INFORMATION &&
          c >= nav2_costmap_2d::INSCRIBED_INFLATED_OBSTACLE) {
        return false;
      }
    }
  }
  return true;
}

bool Explore::selectGoal(const frontier_exploration::Frontier& frontier,
                         const geometry_msgs::msg::Point& robot,
                         geometry_msgs::msg::Point& goal)
{
  nav2_costmap_2d::Costmap2D* costmap = costmap_client_.getCostmap();
  std::lock_guard<nav2_costmap_2d::Costmap2D::mutex_t> lock(
      *(costmap->getMutex()));
  const double res = costmap->getResolution();

  // Frontier cells, closest to the robot first
  std::vector<geometry_msgs::msg::Point> seeds = frontier.points;
  if (seeds.empty()) {
    seeds.push_back(frontier.middle);
  }
  auto dist2 = [&robot](const geometry_msgs::msg::Point& p) {
    return (p.x - robot.x) * (p.x - robot.x) + (p.y - robot.y) * (p.y - robot.y);
  };
  std::sort(seeds.begin(), seeds.end(),
            [&dist2](const geometry_msgs::msg::Point& a,
                     const geometry_msgs::msg::Point& b) {
              return dist2(a) < dist2(b);
            });

  // Try ~10 seeds spread along the frontier (keeps the cost bounded)
  constexpr size_t kMaxSeeds = 10;
  const size_t stride = std::max<size_t>(1, seeds.size() / kMaxSeeds);
  for (size_t i = 0; i < seeds.size(); i += stride) {
    const auto& p = seeds[i];
    const double dx = robot.x - p.x;
    const double dy = robot.y - p.y;
    const double d = std::hypot(dx, dy);
    if (d < 1e-3) {
      continue;
    }
    const double ux = dx / d;
    const double uy = dy / d;
    const double max_s = std::min(max_goal_backoff_, d);
    // Walk from the frontier cell back towards the robot
    for (double s = res; s <= max_s; s += res) {
      // (d - s) = distance from this point to the robot. Goals closer than
      // min_goal_distance_ would be "reached" instantly (xy_goal_tolerance)
      // and only make the robot turn on the spot.
      if (d - s < min_goal_distance_) {
        break;
      }
      const double wx = p.x + ux * s;
      const double wy = p.y + uy * s;
      unsigned int mx, my;
      if (!costmap->worldToMap(wx, wy, mx, my)) {
        break;
      }
      const unsigned char c = costmap->getCost(mx, my);
      // crossed an obstacle: the goal would not see this frontier cell
      if (c != nav2_costmap_2d::NO_INFORMATION &&
          c >= nav2_costmap_2d::INSCRIBED_INFLATED_OBSTACLE) {
        break;
      }
      if (hasClearance(wx, wy, goal_clearance_)) {
        goal.x = wx;
        goal.y = wy;
        goal.z = 0.0;
        return true;
      }
    }
  }
  return false;
}

bool Explore::goalStillUseful(
    const std::vector<frontier_exploration::Frontier>& frontiers)
{
  if (goalOnBlacklist(current_goal_)) {
    return false;
  }
  // The goal was placed at most max_goal_backoff_ from a frontier cell. If no
  // frontier cell is left around it, what it was meant to reveal is known now.
  const double reach = max_goal_backoff_ + 0.5;
  for (const auto& f : frontiers) {
    for (const auto& p : f.points) {
      if (std::hypot(p.x - current_goal_.x, p.y - current_goal_.y) < reach) {
        return true;
      }
    }
  }
  return false;
}

void Explore::finishExploration(const char* reason)
{
  RCLCPP_WARN(logger_, "Exploration complete: %s", reason);
  auto status_msg = explore_lite_msgs::msg::ExploreStatus();
  status_msg.status = explore_lite_msgs::msg::ExploreStatus::EXPLORATION_COMPLETE;
  status_pub_->publish(status_msg);
  stop(true);
}

void Explore::returnToInitialPose()
{
  RCLCPP_INFO(logger_, "Returning to initial pose.");
  auto status_msg = explore_lite_msgs::msg::ExploreStatus();
  status_msg.status = explore_lite_msgs::msg::ExploreStatus::RETURNING_TO_ORIGIN;
  status_pub_->publish(status_msg);

  auto goal = nav2_msgs::action::NavigateToPose::Goal();
  goal.pose.pose.position = initial_pose_.position;
  goal.pose.pose.orientation = initial_pose_.orientation;
  goal.pose.header.frame_id = costmap_client_.getGlobalFrameID();
  goal.pose.header.stamp = this->now();

  auto send_goal_options =
      rclcpp_action::Client<nav2_msgs::action::NavigateToPose>::SendGoalOptions();
  send_goal_options.result_callback =
      [this](const NavigationGoalHandle::WrappedResult& result) {
        if (result.code == rclcpp_action::ResultCode::SUCCEEDED) {
          auto status_msg = explore_lite_msgs::msg::ExploreStatus();
          status_msg.status = explore_lite_msgs::msg::ExploreStatus::RETURNED_TO_ORIGIN;
          status_pub_->publish(status_msg);
          RCLCPP_INFO(logger_, "Successfully returned to initial pose.");
        }
      };
  move_base_client_->async_send_goal(goal, send_goal_options);
}
bool Explore::goalOnBlacklist(const geometry_msgs::msg::Point& goal)
{
  // Euclidean radius in metres (old code: 5 cells = 0.25 m box, smaller than
  // the jitter of a frontier centroid, so blacklisted frontiers came back).
  for (auto& frontier_goal : frontier_blacklist_) {
    if (std::hypot(goal.x - frontier_goal.x, goal.y - frontier_goal.y) <
        blacklist_radius_) {
      return true;
    }
  }
  return false;
}

void Explore::reachedGoal(const NavigationGoalHandle::WrappedResult& result,
                          const geometry_msgs::msg::Point& frontier_goal,
                          const geometry_msgs::msg::Point& frontier_centroid)
{
  // discard stale callbacks from previously preempted goals
  if (result.goal_id != active_goal_id_) {
    return;
  }

  goal_active_ = false;
  switch (result.code) {
    case rclcpp_action::ResultCode::SUCCEEDED:
      RCLCPP_DEBUG(logger_, "Goal was successful");
      // If the frontier is still there after we reached this goal, it cannot
      // be observed from here (pocket behind an object, hole in a wall...).
      // Blacklisting the reached goal stops the old endless loop
      // "send same goal -> instantly succeeded -> send same goal".
      frontier_blacklist_.push_back(frontier_goal);
      has_goal_ = false;
      last_progress_ = this->now();
      prev_distance_ = 0;
      break;
    case rclcpp_action::ResultCode::ABORTED:
#ifdef NAV2_RESULT_HAS_ERROR_CODE
      if (result.result && result.result->error_code != 0) {
        RCLCPP_DEBUG(logger_, "Goal aborted with error_code=%d (%s) — blacklisting frontier",
                     result.result->error_code,
                     result.result->error_msg.c_str());
        frontier_blacklist_.push_back(frontier_goal);
        frontier_blacklist_.push_back(frontier_centroid);
        has_goal_ = false;
      } else {
        RCLCPP_DEBUG(logger_, "Goal aborted with error_code=0 — likely a preemption, not blacklisting");
      }
#else
      // Humble: no error_code field, blacklist unconditionally on abort
      RCLCPP_WARN(logger_, "Nav2 aborted goal (%.2f, %.2f) — blacklisting it",
                  frontier_goal.x, frontier_goal.y);
      frontier_blacklist_.push_back(frontier_goal);
      frontier_blacklist_.push_back(frontier_centroid);
      has_goal_ = false;
#endif
      // If it was aborted probably because we've found another frontier goal,
      // so just return and don't make plan again
      return;
    case rclcpp_action::ResultCode::CANCELED:
      RCLCPP_DEBUG(logger_, "Goal was canceled");
      // If goal canceled might be because exploration stopped from topic. Don't make new plan.
      return;
    default:
      RCLCPP_WARN(logger_, "Unknown result code from move base nav2");
      break;
  }
  // find new goal immediately regardless of planning frequency.
  // Because of the 1-thread-executor nature of ros2 a timer is not needed.
  makePlan();
}

void Explore::start()
{
  RCLCPP_INFO(logger_, "Exploration started.");
  auto status_msg = explore_lite_msgs::msg::ExploreStatus();
  status_msg.status = explore_lite_msgs::msg::ExploreStatus::EXPLORATION_STARTED;
  status_pub_->publish(status_msg);
}

void Explore::stop(bool finished_exploring)
{
  RCLCPP_INFO(logger_, "Exploration stopped.");

  goal_active_ = false;
  // After a pause (e.g. mission node investigating an animal) the goal is
  // treated as new on resume, so the progress timer restarts instead of the
  // frontier being blacklisted straight away.
  has_goal_ = false;
  // Only publish paused status if manually stopped (not finished exploring)
  if (!finished_exploring) {
    auto status_msg = explore_lite_msgs::msg::ExploreStatus();
    status_msg.status = explore_lite_msgs::msg::ExploreStatus::EXPLORATION_PAUSED;
    status_pub_->publish(status_msg);
  }

  move_base_client_->async_cancel_all_goals();
  exploring_timer_->cancel();

  if (return_to_init_ && finished_exploring) {
    returnToInitialPose();
  }
}

void Explore::resume()
{
  resuming_ = true;
  RCLCPP_INFO(logger_, "Exploration resuming.");
  auto status_msg = explore_lite_msgs::msg::ExploreStatus();
  status_msg.status = explore_lite_msgs::msg::ExploreStatus::EXPLORATION_IN_PROGRESS;
  status_pub_->publish(status_msg);
  // Reactivate the timer
  exploring_timer_->reset();
  // Resume immediately
  makePlan();
}

}  // namespace explore

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  // ROS1 code
  /*
  if (ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME,
                                     ros::console::levels::Debug)) {
    ros::console::notifyLoggerLevelsChanged();
  } */
  rclcpp::spin(
      std::make_shared<explore::Explore>());  // std::move(std::make_unique)?
  rclcpp::shutdown();
  return 0;
}