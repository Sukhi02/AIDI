
url='ftp.smartfasal.in'
username = 'testuser@smartfasal.in'
pwd = 'fasal@thapar'

import os 
os.chdir("C:/Users/Raju/Downloads")

import ftplib
print("Fetching libraries")

ftp = ftplib.FTP(url, username, pwd)

print("Acessing FTP server")

#List the files in the current directory
print ("File List:")
files = ftp.dir()

print (files)
ftp.cwd("/")
filename = 'S_AgriB.csv'


print(filename + "file starts downloading")

#Get the readme file



my_file = open(filename, 'wb') # Open a local file to store the downloaded file
print( filename + " File Accesed")
ftp.retrbinary('RETR ' + filename, my_file.write) # Enter
print("\n" + filename + " downloaded")
ftp.quit() # Terminate the FTP connection
print("\nFTP connection terminated succesfully")
my_file.close() # Close the local file you had opened for do




import matplotlib.pyplot as plt 
import pandas as pd

data = pd.read_csv(filename, names=['Timestamp', 'S_M_10cm','S_M_45cm','S_M_80cm', 'Temperature', 'Humidity', 'Pressure', 'Luxes', 'Battery','Readings','Day', 'Date', 'Time'])
# line 1 points 

x1 = data["Timestamp"]
y1 =  data["S_M_10cm"]
print("Plotting begins")
plt.plot(x1, y1, label = "Soil Moisture at 1.5 cm") 
plt.ylabel('Soil Moisture') 
plt.xlabel('Time') 
plt.title('Moisture of the day') 
plt.legend() 
plt.savefig("Soil Moisture at 1(dot)5 cm")
plt.show() 
print("1. Soil Moisture at 15 cm graph plotted \n")

y2 =  data["S_M_45cm"]
plt.plot(x1, y2, label = "Soil Moisture at 4.5 cm") 
plt.ylabel('Soil Moisture') 
plt.xlabel('Time') 
plt.title('Moisture of the day') 
plt.legend() 
plt.savefig("Soil Moisture at 4(dot)5 cm")
plt.show() 
print("2. Soil Moisture at 4.5 cm graph plotted \n")

y3 =  data["S_M_80cm"]
plt.plot(x1, y2, label = "Soil Moisture at 8(dot)0 cm") 
plt.ylabel('Soil Moisture') 
plt.xlabel('Time') 
plt.title('Moisture of the day') 
plt.legend() 
plt.savefig("Soil Moisture at 8(dot)0 cm")
plt.show() 
print("3. Soil Moisture at 8.0 cm graph plotted \n")

y4 =  data["Temperature"]
plt.plot(x1, y4, label = "Temperature") 
plt.ylabel('Atmospheric Temperature') 
plt.xlabel('Time') 
plt.title('Temp of the day') 
plt.legend() 
plt.savefig("Temperature")
plt.show() 
print("4. Temperature graph plotted \n")

y5 =  data["Humidity"]
plt.plot(x1, y5, label = "Humidity") 
plt.ylabel('Humidity') 
plt.xlabel('Time') 
plt.title('Humidity of the Day') 
plt.legend() 
plt.savefig("Humidity")
plt.show() 
print("5. Humidity graph plotted \n")


y6 =  data["Pressure"]
plt.plot(x1, y6, label = "Pressure") 
plt.ylabel('Pressure') 
plt.xlabel('Time') 
plt.title('Pressure of the Day') 
plt.legend()
plt.savefig("Pressure") 
plt.show() 
print( "6. Pressure graph plotted \n ")



y7 =  data["Luxes"]
plt.plot(x1, y7, label = "Luxes") 
plt.ylabel('Luminisity') 
plt.xlabel('Time') 
plt.title('Luminisity of the Day ') 
plt.legend() 
plt.savefig("'Luminisity")
plt.show() 
print("7.  Luxes graph plotted \n ")



data.to_csv(filename + ".csv")
print("8. " + filename + " file is rewritten succesfully \n")

print("code completed")
print("Check your download folder")
