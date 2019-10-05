import paho.mqtt.publish as publish
 
MQTT_SERVER = "192.168.43.53"
MQTT_PATH = "test_channel"
c=str(5)+','+str(8) 
publish.single(MQTT_PATH, c, hostname=MQTT_SERVER)

