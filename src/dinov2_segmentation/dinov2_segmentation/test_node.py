import rclpy
from rclpy.node import Node

class TestNode(Node):
    def __init__(self):
        super().__init__('dinov2_test_node')
        self.get_logger().info('DINOv2 workspace setup successful!')
        
        # 간단한 타이머로 동작 확인
        self.timer = self.create_timer(2.0, self.timer_callback)
    
    def timer_callback(self):
        self.get_logger().info('DINOv2 node is running...')

def main(args=None):
    rclpy.init(args=args)
    node = TestNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()