from PyQt5.QtCore import Qt

class Middleware:
    _instance = None  # Singleton instance

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Middleware, cls).__new__(cls)
            cls._instance.init()
        return cls._instance

    def init(self):
        self.buttonFunctionMap = {}  # Maps buttons to their function pairs (true_func, false_func)
        self.booleanToggle = None 
    
    def setToggle(self, state):
        """
        Set the global toggle controlling the routing.
        :param toggle: QCheckBox or similar UI element
        """
        if state == Qt.Checked:
            self.booleanToggle = True
        else:
            self.booleanToggle = False

    def registerButton(self, button, local_func, ssh_func):
        """
        Register a button and map it to two functions.
        :param button: QPushButton instance
        :param true_func: Function to execute when the toggle is True
        :param false_func: Function to execute when the toggle is False
        """
        self.buttonFunctionMap[button] = (local_func, ssh_func)

    def handleButtonPress(self, button):
        """
        Middleware handler to execute the correct function based on the toggle state.
        :param button: QPushButton instance
        """
        if button in self.buttonFunctionMap:
            local_func, ssh_func = self.buttonFunctionMap[button]
            if self.booleanToggle and self.booleanToggle.checked:
                local_func()
            else:
                ssh_func()
        else:
            print(f"No mapping found for button: {button.text()}")
