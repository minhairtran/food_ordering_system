There are 3 actors: Customer, robot and staff:
- Customer is the actor coming to restaurant and order food with robot.
- Robot is the actor chatting with customer and send order requests to staff.
- Staff is the actor managing the operation of the application and managing the order requests.

![Phân tích hệ thống-tổng quan](https://user-images.githubusercontent.com/49912069/123767691-dd830000-d8f1-11eb-9b30-a7cacdf4ff94.png)

Business specifications:
- *Start the system*
  - Objective: For starting the application.
  - Primary actor: Staff
  - Trigger: Runing the command "python application.py" or "python3 application.py" in terminal
  - Preconditions: The application is off and the device for running the application was configured with libraries in [requirements.txt](https://github.com/minhairtran/food_ordering_system/blob/main/requirements) and [python](https://www.python.org/downloads/)

- *Start conversation*
  - Objective: For customer to start ordering with robot.
  - Primary actor: Staff
  - Other participating actor: Customer
  - Trigger: Staff clicks checkbox or presses shortcut Ctrl + x, of which value is equal to the table value at which the customer is sitting
  - Preconditions: The application is running and there's a new customer coming to a seat (Staff may notice this by watching camera in the restaurants)
  - Primary screen:
    - Staff clicks checkbox or presses shortcut Ctrl + x
    - System starts the conversation for the new customer

- *Order with robot*
  - Objective: For customer to order food with robot.
  - Primary actor: Customer and robot
  - Preconditions: Conversation has been started
  - Primary screen:
    - Robot talks by means of the speaker and the customer talks by means of the microphone, all placed at the table, with the following screens:
![Phân tích hệ thống-User experience](https://user-images.githubusercontent.com/49912069/123767201-80874a00-d8f1-11eb-8b21-2f2aa7161672.png)

- *Save customer record*
  - Objective: Save customer record is for further training
  - Primary actor: Robot and customer
  - Trigger: When the conversation starts
  - Preconditions: 
    - The conversation starts
    - There's at least one time customer says "không" (no) during the conversation
  - Primary screen:
    - Robot records customer
    - Customer says "không"
    - At the end of the conversation, the record is saved
  - Secondary screen:
    - Robot records customer
    - Customer doesn't say "không"
    - At the end of the conversation, the record isn't saved

- *Display food order list*
  - Objective: For staff to get the food orders 
  - Primary actor: Robot
  - Trigger: After a conversation between robot and customer finishes
  - Preconditions: The application is running 
  - Primary screen: 
    - Robot adds the food ordered and the table number on top of order list
  - Secondary screen:
    - In case robot fails to help customer order, the table number followed by "nhân viên" (staff) is added on top of order list

UI design

![image](https://user-images.githubusercontent.com/49912069/123766982-56358c80-d8f1-11eb-9e77-21468a4a4858.png)
