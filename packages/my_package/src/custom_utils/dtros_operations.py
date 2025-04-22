import rospy

class DtrosOperations:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DtrosOperations, cls).__new__(cls)
        return cls._instance

    @staticmethod
    def unregister_and_delete_subscribers(task):
        for attr_name in dir(task):
            if attr_name.startswith("_sub_"):
                attr = getattr(task, attr_name)
                if hasattr(attr, "unregister") and callable(attr.unregister):
                    try:
                        attr.unregister()
                        # print(f"Unregistered {attr_name}")
                        # delattr(task, attr_name)
                        # print(f"Deleted {attr_name}")
                    except Exception as e:
                        rospy.logwarn(f"Failed to unregister {attr_name}: {e}")
    
    @staticmethod
    def unregister_subscribers(subscribers):
        if not isinstance(subscribers, dict):
            raise TypeError("Expected a dictionary of subscribers, but got: " + str(type(subscribers)))
        for name, sub in subscribers.items():
            if not isinstance(name, str):
                raise ValueError(f"Subscriber name must be a string, but got: {type(name)}")

            if not hasattr(sub, 'unregister') or not callable(sub.unregister):
                raise ValueError(f"Value for key '{name}' is not a valid rospy.Subscriber or doesn't have an 'unregister' method")
            if sub is not None:
                try:
                    sub.unregister()
                    rospy.loginfo(f"Unregistered subscriber: {name}")
                except Exception as e:
                    rospy.logwarn(f"Failed to unregister subscriber {name}: {e}")

    @staticmethod
    def runWithNoAction():
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            rate.sleep()