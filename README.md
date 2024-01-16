# Leaf_Image_Retrieval_and_Recognition_MobileApp_FYP

## Overview

This project aims to develop a Leaf Image Retrieval and Recognition System, comprising a mobile application and a web-based system for administrators. The primary objective is to address the challenges faced in leaf identification and retrieval, replacing manual methods with efficient visual processing and machine learning techniques.

## Getting Started

### Prerequisites

- [Visual Studio Code (VSC)](https://code.visualstudio.com/) or any preferred IDE
- [Python](https://www.python.org/) installed
- Obtain the correct IP address of the server or device for network requests.

Instructions:
Check your IP address: Make sure you know the IP address of the server or device where you want to send the network request.

Update the IP address: Replace the IP address "192.168.0.155" in the code snippet with the correct IP address you obtained in step 1.

Starting the server:
1. Open the Web App folder with an IDE such as Visual Studio Code (VSC).
2. Open the terminal within the IDE.
3. Make sure that you are in the correct directory within the terminal. You should be in the same directory where the app.py file is located. You can use the cd command to       navigate to the correct directory if needed.
4. Once you are in the correct directory, enter the command python app.py in the terminal and press Enter.
5. The server should start up, and you should see any relevant output or logs in the terminal. You may also see messages indicating that the server is running and listening on a specific port.
6. You can now access the server and interact with the web application through your web browser. In most cases, you can visit http://localhost:port (replace "port" with the actual port number specified in your app.py file) to access the application.



----------------------
Code Readme
----------------------

This code snippet can be found in the file SelectedPhotoActivity.java. It is located at line 217.

----------------------
Code Description:
----------------------

The purpose of this code snippet is to perform a network request to process an image using a specified IP address. You need to ensure that you have the correct IP address for your network environment.

----------------------
Code Snippet:
----------------------

```java
check yr ip address and change the current address to your ip address

Current Address:
Request request = new Request.Builder()
                    .url("http://192.168.0.155:5000/process_image")


Example: the new IP address is 192.168.0.100
New Address:
Request request = new Request.Builder()
                    .url("http://192.168.0.100:5000/process_image")
