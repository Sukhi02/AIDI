import paho.mqtt.publish as publish
import Adafruit_DHT
import RPi.GPIO as g
import Adafruit_GPIO.SPI as SPI
import Adafruit_MCP3008
from time import sleep
import os
import paho.mqtt.client as mqtt

a=0
print("sleeping ")
sleep(10)

g.setmode(g.BCM)
#g.setup(4, g.IN)
g.setup(17, g.IN)
g.setup(22, g.OUT)
g.setup(27, g.OUT)

CLK  = 11
MISO = 9
MOSI = 10
CS   = 8
mcp = Adafruit_MCP3008.MCP3008(clk=CLK, cs=CS, miso=MISO, mosi=MOSI)

MQTT_SERVER = "192.168.43.53"
MQTT_PATH = "test"
MQTTNODE2="192.168.43.144"
for i in os.listdir('/sys/bus/w1/devices'):
        if i != 'w1_bus_master1':
            ds18b20 = i

def on_connect(client, userdata, flags, rc):
    print("Connected with result code "+str(rc))

    # Subscribing in on_connect() means that if we lose the connection and
    # reconnect then subscriptions will be renewed.
    client.subscribe(MQTT_PATH)

def on_message(client, userdata, msg):
    global a
    a = str(msg.payload)

client=mqtt.Client()
client.on_connect=on_connect
client.on_message=on_message
client.connect(MQTTNODE2, 1883, 60)

while True:
        location = '/sys/bus/w1/devices/' + ds18b20 + '/w1_slave'
        tfile = open(location)
        text = tfile.read()
        tfile.close()
        secondline = text.split("\n")[1]
        temperaturedata = secondline.split(" ")[9]
        temperature = float(temperaturedata[2:])
        celsius = temperature / 1000

        #moist=g.input(4)

        client.loop_start()
        client.loop_stop()
        '''if(a==1):
            g.output(22, 1)
            g.output(27, 0)
            sleep(5)
            g.output(22, 0)
            g.output(27, 0)'''

        values = mcp.read_adc(0)
        dat=100-(100*values/1023)
	print("a==", a)
        if(dat<35 or a=='1'):
                print("no water detected")
                g.output(22, 1)
                g.output(27, 0)
		if(dat<35):
			print("no water detected")
		else:
			print("water detected")
        else:
		g.output(22, 0)
                g.output(27, 0)
                print("water detected")
        sleep(2)
        #g.output(22,0)
        #g.output(27, 0)
        humd, temp = Adafruit_DHT.read_retry(11, 17)
        print("soil moist = ",dat)
	print("humidity = ", humd)
	print("temperature = ", temp)
	print("soil temp = ", celsius)
        v=str(1)+','+str(humd)+','+str(temp)+','+str(dat)+','+str(celsius)
        publish.single(MQTT_PATH, v, hostname=MQTT_SERVER)
        #print("1")

