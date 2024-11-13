    **Stock Market Learning Portal**
                                                                          
Overview:
The Stock Market Learning Portal is a comprehensive educational platform aimed at helping users learn about the stock market, from basic concepts to advanced trading strategies. 
The portal provides an intuitive interface, user profiles, and various resources to help users get started in stock market trading.

This web-based platform includes sections on stock market fundamentals, trading basics, advanced concepts, and learning resources, along with user login and profile management. 
The portal is designed using HTML, CSS, JavaScript (for interactivity), and Flask for the backend.


Features:
User Authentication: Users can log in and view personalized information such as their username, email, and phone number. Logged-in users can log out via the sidebar.
Sidebar Navigation: A collapsible sidebar for quick access to key sections of the platform.
Learning Sections: Interactive sections on stock market fundamentals, types of stocks, market participants, order types, technical and fundamental analysis, trading strategies, and more.
Resources: A curated list of online resources and tools for further learning and practice.
Responsive Design: The platform is mobile-friendly, using Bootstrap to ensure it adapts to different screen sizes.


Installation:
1. Prerequisites
2. Python 3.x
3. Flask (Python framework)
4. SQLite for storing user data (via Flask)

Setup Instructions:
1. Clone this repository to your local machine:
  git clone https://github.com/HARSHALCHAUDHARI/Project.git
2. Navigate to the project folder:
  cd stock-market-learning-portal
3. Install the required Python dependencies:
  pip install -r requirements.txt
4. Set up the SQLite database:
  The Flask app will automatically create a database.db file when the user logs in for the first time.
5. Run the Flask app:
  python app.py
6. Open the application in your browser:
  http://127.0.0.1:5000




How It Works: 

* User Authentication: The app allows users to sign up and log in using a username and password. When logged in, user information (e.g., email, phone number) is displayed in the sidebar.
* Sidebar: The sidebar provides quick access to different sections of the website (introduction, stock basics, advanced concepts, resources, and profile management). It is collapsible for    better usability on mobile devices.
* Learning Sections: The website is divided into various learning modules, each explaining key concepts like stock types, market participants, technical analysis, and trading strategies.
* Logout: The logout button allows users to log out, which clears their session.

  
Technologies Used:
1. HTML/CSS: Used for the structure and styling of the portal.
2. Bootstrap: A CSS framework for responsive design.
3. Flask: A Python web framework for building the server-side logic and handling user authentication.
4. SQLite: A lightweight database used to store user information.
5. JavaScript: Used for client-side interactivity (e.g., sidebar toggle).


Resources: 
Here are some external learning resources included in the platform:
* 5paisa Learning Center
* Investopedia
* MoneyControl
* Virtual Trading Simulator (to practice trading without risk)

Contributing: 
Contributions are welcome! If you have suggestions or want to report bugs, feel free to open an issue or submit a pull request.
* Fork the repository.
* Create a new branch (git checkout -b feature-name).
* Commit your changes (git commit -am 'Add new feature').
* Push to the branch (git push origin feature-name).
* Create a new Pull Request.

Acknowledgments:
  * Bootstrap: For the responsive design and UI components.
  * Font Awesome: For providing the icons used in the sidebar.
  * Flask: For simplifying the development of the backend.

Screenshots: 
![image](https://github.com/user-attachments/assets/9a38d828-f7fc-461a-808f-3a83cab2080a)
