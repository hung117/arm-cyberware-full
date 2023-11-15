sudo chmod 666 /dev/ttyUSB0
sudo chmod 666 /dev/rfcomm0
what to do next ?
bluetooth
connect to bluetooth with arduino:

hcitool scan
get the MAC

sudo rfcomm bind 0 98:DA:60:07:9E:8D 1 // bind with MAC, 0:device, 1: channel
edit the etc/bluetooth/rfcomm.conf:
```
rfcomm0 {
	# Automatically bind the device at startup
	bind no;

	# Bluetooth address of the device
	device 98:DA:60:07:9E:8D;

	# RFCOMM channel for the connection
	channel    1;

	# Description of the connection
	comment "HC-06";
    }

```
sudo minicom -D /dev/rfcomm0 // wait for a while and the connection will be set

then connect accordingly in python

to release:
sudo rfcomm release 0

improve pose with millis, change each finger gradually, to counter hardware limitation
uno rx,tx pin, software Serial
serial communication