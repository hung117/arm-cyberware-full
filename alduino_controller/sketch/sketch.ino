#include <Servo.h>
#include <SoftwareSerial.h>
#include <string.h>
using namespace std;
// for nano board
Servo fin_middle; // create servo object to control middle finger
Servo fin_idx;    // create servo object to control index finger
Servo fin_thumb;  // create servo object to control index finger

int number_of_poses = 7;
// int pin_mode = 3;
int pin_mode = 4;
int pin_pose = 7;
const int pin_bluetooth = 12;
// int potpin_mid = 0;   // analog pin used to connect the potentiometer
// int potpin_idx = 1;   // analog pin used to connect the potentiometer
int potpin_mid = 3;   // analog pin used to connect the potentiometer
int potpin_idx = 4;   // analog pin used to connect the potentiometer
int potpin_thumb = 2; // analog pin used to connect the potentiometer

int val_mid;   // variable to read the value from the analog pin
int val_idx;   // variable to read the value from the analog pin
int val_thumb; // variable to read the value from the analog pin

// int speed = 19200;
int speed = 9600;

int old_val_idx;
int old_val_mid;
int old_val_thumb;
// BLUE TOOTH SECTION
// const int pin_Rx = 3;                    // receive
// const int pin_Tx = 2;                    // transmit
const int pin_Rx = 2;                    // receive
const int pin_Tx = 3;                    // transmit
SoftwareSerial BTSerial(pin_Rx, pin_Tx); //==
// the physics rx,tx on UNO is for communicating with pc, monitoring, updating n stuffs
void setup()
{
  // BTSerial.begin(speed);
  BTSerial.begin(9600);
  // BTSerial.begin(speed);
  Serial.begin(speed);
  pinMode(pin_mode, INPUT);
  pinMode(pin_pose, INPUT);
  pinMode(pin_bluetooth, INPUT);
  pinMode(pin_Rx, INPUT);
  pinMode(pin_Tx, OUTPUT);

  fin_middle.attach(9); // attaches the servo on pin 9 to the servo object
  fin_idx.attach(11);   // attaches the servo on pin 11 to the servo object
  fin_thumb.attach(10); // attaches the servo on pin 11 to the servo object

  old_val_mid = 0;
  old_val_idx = 0;
  old_val_thumb = 0;
}
void onchange_monitor(int val_old, int val_new, String name = "_name")
{
  if (val_old < val_new - 5 || val_old > val_new + 5)
  {
    Serial.print(name);
    Serial.println(val_new, DEC);
    val_old = val_new;
  }
}
bool b_mannual = true;
bool b_bluetooth = false;
int b_oldbuttonState = 0;
void mannualControl()
{
  val_mid = analogRead(potpin_mid);        // reads the value of the potentiometer (value between 0 and 1023)
  val_mid = map(val_mid, 0, 1023, 0, 180); // scale it to use it with the servo (value between 0 and 180)
  fin_middle.write(val_mid);
  // onchange_monitor(old_val_mid, val_mid, "fin_mid ");

  val_idx = analogRead(potpin_idx);        // reads the value of the potentiometer (value between 0 and 1023)
  val_idx = map(val_idx, 0, 1023, 0, 180); // scale it to use it with the servo (value between 0 and 180)
  fin_idx.write(val_idx);                  // sets the servo position according to the scaled value
  // onchange_monitor(old_val_idx, val_idx, "fin_idx ");

  val_thumb = analogRead(potpin_thumb);        // reads the value of the potentiometer (value between 0 and 1023)
  val_thumb = map(val_thumb, 0, 1023, 0, 180); // scale it to use it with the servo (value between 0 and 180)
  fin_thumb.write(val_thumb);                  // sets the servo position according to the scaled value
  // onchange_monitor(old_val_thumb, val_thumb, "fin_thumb ");

  delay(15);
}
int i_pose = 0; // 4 pose atm 0 - 1 - 2 - 3
void updatePose()
{
  // get mode pose
  int buttonState = digitalRead(pin_mode);

  if (buttonState == HIGH)
  // if (buttonState != b_oldbuttonState)
  {
    b_mannual = true;
  }
  else
  {
    b_mannual = false;
  }

  buttonState = digitalRead(pin_bluetooth);
  if (buttonState == HIGH)
  {
    b_bluetooth = true;
  }
  else
  {
    b_bluetooth = false;
  }

  int poseState = digitalRead(pin_pose);
  if (poseState == HIGH)
  {
    if (i_pose < number_of_poses)
    {
      i_pose += 1;
    }
    else
    {
      i_pose = 0;
    }
  }
  delay(5);
}
bool b_InitPose = false;
int angle_fin = false;
// Define finger motion here with angle & delay time (angle,t_wait)
void moveFinger(Servo& finger,int angle, int t_wait){
  delay(t_wait);
  finger.write(angle);
}
void PoseChange()
{
  // 1 close all
  // 2 close 2fin - thumb & index (slight)
  // 3 close all except thumb
  // 4 close 3 fin thumb & index & middle
  // 5 close all (50%)
  // 6 close all
  switch (i_pose)
  {
  case 0: // Palm Open
    moveFinger(fin_thumb,180,10);
    moveFinger(fin_idx,180,0);
    moveFinger(fin_middle,0,0);
    break;
  case 1: // close all
    moveFinger(fin_thumb,0,10);
    moveFinger(fin_idx,0,0);
    moveFinger(fin_middle,180,0);

    break;
  case 2: // close 2 fin - thumb & index (slight)
    moveFinger(fin_thumb,0,10);
    moveFinger(fin_idx,180,0);
    moveFinger(fin_middle,0,0);

    break;
  case 3: // close all except thumb
    moveFinger(fin_thumb,180,10);
    moveFinger(fin_idx,0,0);
    moveFinger(fin_middle,180,0);

    break;
  case 4: // close 3 fin thumb & index & middle
    moveFinger(fin_thumb,0,20);
    moveFinger(fin_idx,0,0);
    moveFinger(fin_middle,180,0);

    break;
  case 5: //close all (50%)
    moveFinger(fin_thumb,45,10);
    moveFinger(fin_idx,90,0);
    moveFinger(fin_middle,90,0);

    break;
  case 6: // close 3 fin thumb & index & middle
    moveFinger(fin_thumb,50,20);
    moveFinger(fin_idx,50,0);
    moveFinger(fin_middle,50,0);

    break;
  case 7: // point
    fin_thumb.write(0);
    fin_idx.write(180); 
    fin_middle.write(180);

    break;
  default: // Open
    fin_thumb.write(180);
    fin_idx.write(180); 
    fin_middle.write(0);
    
    break;
  }
}
void print_mode()
{
  if(b_bluetooth){
    Serial.print("     bluetooth: ");
    Serial.print(b_bluetooth);
    Serial.print(" ---- ");
    Serial.print("pose: ");
    Serial.println(i_pose);
  }else{
    Serial.print("mannual ");
    Serial.print(b_mannual);
    String fin_vals = " -- thumb: " + String(val_thumb) + " idx: " + String(val_idx) + " mid: " + String(val_mid);
    Serial.print(fin_vals);
    Serial.print(" ---- ");
    Serial.print("pose: ");
    Serial.println(i_pose);
  }


  delay(200);
}
String messageBuffer = "";
String message = "";
void loop()
{
  updatePose();
  print_mode();

  if (b_bluetooth)
  { 
    PoseChange();
    if (BTSerial.available() > 0)
    {
      // Serial.print("available ");
      // Serial.println(BTSerial.available() > 0);
      char data = (char)BTSerial.read();
      messageBuffer += data;
      if (data == ';')
      {
        message = messageBuffer;
        messageBuffer = "";
        Serial.print(message);
        // String msg_send = "received: " + message;
        // BTSerial.print(message);
        // BTSerial.print(msg_send);
        i_pose = message.substring(0).toInt();
        Serial.print(i_pose);
        BTSerial.print(i_pose);
        // PoseChange();
      }
    }
  }
  else
  {
    // updatePose();
    if (b_mannual)
    {
      mannualControl();
    }
    else
    {
      PoseChange();
    }
  }
}
