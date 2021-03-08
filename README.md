# TheEmergencyHelper
Security system for bulidings to monitor incoming andoutgoing traffic in real time using machine learning approach.

Nowadays, many security events occur in buildings where there is no information about the persons who are inside. “Emergency Helper” is an interactive system that can provide information to the security forces about the persons that are in the building in real-time. With this kind of feature, the emergency teams would have more control over the persons in the building in case of an emergency.
In our project, we developed an interactive WEB system, based on big-data algorithms and resources, combined with the machine learning approach. In this paper, we will present the big-data and machine learning that our system uses.
The aim of this work is to recognize the people present in the building or estimate their age and gender if face recognition failed. All results from the system display on a website accessible to any platform from any place. We managed to achieve high results from our system regarding face detection, face recognition, and age & gender estimation.  
The face detection task accomplished using the Viola-Jones algorithm, an algorithm that is especially suitable for real-time tasks. 
For face recognition, we are using the base architecture of the MobileNetV2 (CNN) model and adjust it to our project task. 
In case face recognition failed, the software estimates the age & gender of the person in the image. For the age & gender estimation, we designed a CNN (Convolutional Neural Network) model that predict both, the age and the gender.
Our system established under a flow of eight main steps, as depicted in the EPC (Event-driven Process Chain) diagram below.
At the end of the processes chain, the expected data will be posted to the WEB system.

![image](https://user-images.githubusercontent.com/46426884/110316557-93f56900-8013-11eb-81b3-4c154ff847dc.png)

