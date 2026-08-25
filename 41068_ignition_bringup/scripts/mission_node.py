#!/usr/bin/env python3

import math
from enum import Enum

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
from std_msgs.msg import String, Bool
from std_srvs.srv import Trigger


# ============================================================
# Mission states
# ============================================================

class MissionState(Enum):
    INITIALISING = 0
    COVERAGE = 1
    INVESTIGATING = 2
    COMPLETE = 3


# ============================================================
# Candidate states
# ============================================================

class CandidateState(Enum):
    UNINVESTIGATED = 0
    INVESTIGATING = 1
    CONFIRMED_exotic = 2      
    CONFIRMED_native= 3  
   



class MissionNode(Node):

    def __init__(self):
        super().__init__('mission_node')

        # ---------------- Mission ----------------

        self.state = MissionState.INITIALISING


        # ---------------- Candidate ----------------

        self.candidates = {}
        self.current_candidate = None 
        self.next_candidate_id = 1
        self.candidate_merge_distance = 1.0


        # ---------------- Subscribers ----------------



        self.candidate_sub = self.create_subscription(Point, 'candidate', self._candidate_callback, 10)




        self.result_sub = self.create_subscription(String, 'Yolo_result', self.result_callback, 10)

        # ---------------- Publisher --------------------------------------------------------------------------------------------



        
        self.investigation_target_pub = self.create_publisher(String,'navigation_target',10)

        # puase/ resume package
        self.exploration_resume_pub = self.create_publisher(Bool,'exploration_command',10)






        self.ui_status_pub = self.create_publisher(String,'mission_status', 10)




        self.timer = self.create_timer(0.5, self._mission_tick)

        self.get_logger().info('Mission Node started')


    # ========================================================
    # MISSION STATE MACHINE
    # ========================================================

    def _mission_tick(self):

        #publish information to user interface
        self._publish_ui_status()

        # ----------------------------------------------------
        # INITIALISING
        # ----------------------------------------------------

        if self.state == MissionState.INITIALISING:


            if not self.coverage_received:
                self.get_logger().info('Waiting for coverage points...', throttle_duration_sec=5.0)
                return

            self.get_logger().info('Mission initialised.')
            self.state = MissionState.COVERAGE
            return


        # ----------------------------------------------------
        # COVERAGE
        # ----------------------------------------------------

        if self.state == MissionState.COVERAGE:






            candidate_id = self._get_next_candidate()

            if candidate_id is not None:

                self._pause_exploration()
    
                self.current_candidate = candidate_id
                self.get_logger().info(f'Candidate detected: {candidate_id}')
                self.state = MissionState.INVESTIGATING
                return

            return

        # ----------------------------------------------------
        # INVESTIGATING
        # ----------------------------------------------------

        if self.state == MissionState.INVESTIGATING:


            if self.current_candidate is None:
                self.state = MissionState.COVERAGE
                return

            candidate = self.candidates[self.current_candidate]


            if candidate['state'] == CandidateState.UNINVESTIGATED:
                candidate['state'] = CandidateState.INVESTIGATING
                self._publish_investigation_target(candidate['x'], candidate['y'])
                self.get_logger().info(f'Investigation target sent to R3: {self.current_candidate} at ({candidate["x"]:.2f}, {candidate["y"]:.2f})')
                return



            if candidate['state'] == CandidateState.CONFIRMED_exotic:
                self.get_logger().info(f'{self.current_candidate} confirmed: {candidate["animal_class"]}')
                self.current_candidate = None
  

                self._resume_exploration()
                self.state = MissionState.COVERAGE
                return


            if candidate['state'] == CandidateState.CONFIRMED_native:
                self.get_logger().info(f'{self.current_candidate} rejected: {candidate["animal_class"]}')
                self.current_candidate = None


                self._resume_exploration()
                self.state = MissionState.COVERAGE
                return



            #if candidate['state'] == CandidateState.UNRESOLVED:
              #  self.get_logger().info(f'{self.current_candidate} unresolved.')
              #self.current_candidate = None
              #  self.navigation_status = 'IDLE'
              #  self.state = MissionState.COVERAGE
              #  return


            if candidate['state'] == CandidateState.INVESTIGATING:
                return


        # ----------------------------------------------------
        # COMPLETE
        # ----------------------------------------------------

        if self.state == MissionState.COMPLETE:
            return


    # ========================================================
    # RECEIVE NEW CANDIDATE
    # ========================================================

   


    def _candidate_callback(self, msg: Point):


        x = msg.x
        y = msg.y


        for candidate in self.candidates.values():
            dx = x - candidate['x']
            dy = y - candidate['y']
            distance = math.hypot(dx, dy)

            if distance < self.candidate_merge_distance:
                self.get_logger().info('Candidate ignored because it is too close to an existing candidate.')
                return



        candidate_id = f'candidate_{self.next_candidate_id}'
        self.next_candidate_id += 1



        #dictionary[key] = value

        self.candidates[candidate_id] = {
            'x': x,
            'y': y,
            'state': CandidateState.UNINVESTIGATED,
            'animal_class': None,
            'confidence': None
        }

        self.get_logger().info(f'New candidate stored: {candidate_id} at ({x:.2f}, {y:.2f})')


    # ========================================================
    # FIND NEXT CANDIDATE
    # ========================================================



    def _get_next_candidate(self):


        for candidate_id, data in self.candidates.items():
            if data['state'] == CandidateState.UNINVESTIGATED:
                return candidate_id

        return None


    # ========================================================
    # PUBLISH NAVIGATION TARGET
    # ========================================================



    def _publish_investigation_target(self, x, y):

        msg = Point()
        msg.x = x
        msg.y = y
        msg.z = 0.0

        self.investigation_target_pub.publish(msg)
        self.navigation_status = 'REQUESTED'

        self.get_logger().info( f'Investigation target sent to R3: ({x:.2f}, {y:.2f})'
    )


    # ========================================================
    # INVESTIGATION RESULT CALLBACK
    # ========================================================
 


    def result_callback(self, msg: String):



            animal_class, confidence, is_exotic = msg.data.split(',')


            if self.current_candidate is None:
                self.get_logger().warn(
                'YOLO result received but no candidate is currently being investigated.')
                return



            candidate_id = self.current_candidate
            candidate = self.candidates[candidate_id]

            
            candidate['animal_class'] = animal_class
            candidate['confidence'] = float(confidence)


            if is_exotic.lower() == 'true':
               candidate['state'] = CandidateState.CONFIRMED_exotic 
            else:
               candidate['state'] = CandidateState.CONFIRMED_native
  
            self.get_logger().info( f'{candidate_id}: {animal_class}, confidence={confidence}, exotic={is_exotic}' )
            self.get_logger().info('Investigation result received.')
            
    #======================================================================


    def _resume_exploration(self):

        msg = Bool()
        msg.data = True
  
        self.exploration_resume_pub.publish(msg)

        self.get_logger().info('Resume command sent to exploration package.')


    def _pause_exploration(self):

        msg = Bool()
        msg.data = False

        self.exploration_resume_pub.publish(msg)

        self.get_logger().info('Pause command sent to exploration package.')


    #=============  GUI  information  =================================

    def _publish_ui_status(self):


        candidate_text = []

        exotic_count = 0
        native_count = 0

        for candidate_id, candidate in self.candidates.items():

            candidate_text.append(
                f'{candidate_id}:'
                f'{candidate["state"].name}:'
                f'{candidate["animal_class"]}:'
                f'{candidate["confidence"]}:'
                f'{candidate["x"]}:'
                f'{candidate["y"]}'
            )
  
            if candidate['state'] == CandidateState.CONFIRMED_exotic:
                exotic_count += 1

            elif candidate['state'] == CandidateState.CONFIRMED_native:
                native_count += 1

        msg = String()

        msg.data = (
            f'robot_state={self.state.name};'
            f'exotic_count={exotic_count};'
            f'native_count={native_count};'
            f'candidates={"|".join(candidate_text)}'
        )
        self.ui_status_pub.publish(msg)


#===========================================================================================



def main(args=None):

    rclpy.init(args=args)
    node = MissionNode()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()