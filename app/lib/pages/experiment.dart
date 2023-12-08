// import 'dart:js_interop';

// ignore_for_file: prefer_const_constructors

import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'dart:math';
import 'dart:ui';
import 'package:flutter/material.dart';
import 'package:app/grpc_generated/client.dart';
import 'package:app/grpc_generated/init_py.dart';
import 'package:flutter/scheduler.dart';
// import 'package:app/grpc_generated/service.pbgrpc.dart';
import '../grpc_generated/services.pbgrpc.dart';
import '../custom_widgets/pythonLoaderMonitor.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:image_picker/image_picker.dart';
import 'dart:async';
import 'dart:io';
import 'package:path_provider/path_provider.dart';
import 'dart:convert';
import 'package:flutter/services.dart' show rootBundle;

Future<void> pyInitResult = Future(() => null);
Future<void> pyImportResult = Future(() => null);
bool b_predRun = false;

class Experiment extends StatefulWidget {
  const Experiment({Key? key}) : super(key: key);

  @override
  State<Experiment> createState() => ExperimentState();
}

class ExperimentState extends State<Experiment> {
  List<int> randomIntegers =
      List.generate(40, (index) => Random().nextInt(100));
  int iPose = 0;
  late var bytes;
  late Image _image;

  String base64 = "";

  var default_base64_img_str = "";

  Future<String> loadAsset() async {
    String str = "";
    await rootBundle.loadString('assets/mytext.txt').then((value) {
      str = value;
    });
    return str;
  }

  late Timer _everySecond;

  @override
  void initState() {
    WidgetsFlutterBinding.ensureInitialized();
    pyInitResult = initPy();
    super.initState();

    _everySecond = Timer.periodic(Duration(seconds: 2), (Timer t) {
      setState(() {
        MyFrameCallBack(postFrameDelay);
      });
    });
    getDefaultImage().then((value) => {
          default_base64_img_str = value,
          print(value),
          base64 = default_base64_img_str
        });
  }

  void update() {
    setState(() {
      b_predRun = false;
      iPose += 1;
      if (iPose > 5) {
        iPose = 0;
        getDefaultImage().then((value) => {default_base64_img_str = value});
      }
    });
  }

  String path_str = 'assets/signal.png';
  int sample_idx = 0;

  Future getImage() async {
    // using your method of getting an image
    Image newImage = Image.file(
      File('/storage/emulated/0/Download/test.jpg'),
    );
    setState(() {
      _image = newImage;
    });
  }

  Future<String> getDefaultImage() async {
    var res = "";
    await loadAsset().then((value) {
      print("DONEEEE!");
      res = value;
    }).catchError((err) {
      return " can not load string getDefault img";
    });
    return res;
  }

  bool bPredRunning = false;
  int idxFrom = 0;
  int idxTo = -1;
  Future<void> predict_getData() async {
    bPredRunning = true;
    await EMGClassifierServiceClient(getClientChannel())
        .classify_Signal(PredictRequest(
            idxFrom: idxFrom, idxTestPose: pose_to_predict, idxUser: testUser))
        .then((p0) => {
              sample_idx++,
              iPose = p0.signal,
              base64 = "data:image/jpeg;base64," + p0.base64plot,
              idxFrom++,
              bPredRunning = false
            });
  }

  final postFrameDelay = Duration(seconds: 100);

  void MyFrameCallBack(Duration timestamp) {
    setState(() {
      if (b_predRun && idxFrom < 600 && !bPredRunning) {
        predict_getData();
        print('run pred');
      } else {
        print('stop run pred');
        if (default_base64_img_str != " ") {
          // base64 = default_base64_img_str;
        }
      }
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        backgroundColor: Colors.blueGrey,
        title: const Text('EXPERIMENT',
            style: TextStyle(
                color: Colors.tealAccent,
                fontWeight: FontWeight.w500,
                fontSize: 48)),
        centerTitle: true,
      ),
      body: SingleChildScrollView(
        child: Container(
          color: Colors.lime[100],
          padding: const EdgeInsets.all(20),
          // alignment: Alignment.center,
          width: 1000000,
          child: Column(
            mainAxisAlignment: MainAxisAlignment.start,
            crossAxisAlignment: CrossAxisAlignment.center,
            children: [
              Text("IPOSE: ${iPose}, bPredRun ${b_predRun}"),
              SizedBox(
                height: 50,
                child: PythonLoaderMonitor(),
              ),
              const SizedBox(height: 16),
              ElevatedButton(
                onPressed: () {
                  base64 = default_base64_img_str;
                  update();
                  NumberSortingServiceClient(getClientChannel())
                      .sortNumbers(NumberArray(numbers: randomIntegers))
                      .then(
                          (p0) => setState(() => randomIntegers = p0.numbers));
                },
                style: ElevatedButton.styleFrom(
                  minimumSize:
                      const Size(140, 36), // Set minimum width to 120px
                ),
                child: const Text('debug increase pose mannually'),
              ),
              const SizedBox(height: 16),
              Row(
                children: [
                  ElevatedButton(
                    onPressed: () {
                      setState(() {
                        pyInitResult = initPy();
                      });
                    },
                    style: ElevatedButton.styleFrom(
                      minimumSize:
                          const Size(140, 36), // Set minimum width to 120px
                    ),
                    child: const Text('python server connection retry'),
                  ),
                  ElevatedButton(
                    onPressed: () {
                      setState(() {
                        b_predRun = !b_predRun;
                        update();
                      });
                    },
                    style: ElevatedButton.styleFrom(
                      minimumSize:
                          const Size(140, 36), // Set minimum width to 120px
                    ),
                    child: const Text('signal predict'),
                  ),
                ],
              ),
              Padding(
                padding: const EdgeInsets.all(10.0),
                child: User_pose_selector(),
              ),
              Container(
                  padding: const EdgeInsets.all(16.0),
                  height: 250,
                  width: 400,
                  decoration: BoxDecoration(
                    border: Border.all(
                      width: 5,
                    ),
                    borderRadius: BorderRadius.all(Radius.circular(40)),
                  ),
                  child: Image(image: AssetImage('assets/${iPose}.png'))),
              Container(
                  padding: const EdgeInsets.all(16.0),
                  height: 300,
                  width: 400,
                  decoration: BoxDecoration(
                    border: Border.all(
                      width: 5,
                    ),
                    borderRadius: BorderRadius.all(Radius.circular(40)),
                  ),
                  child: FutureBuilder<String>(
                    future: loadAsset(),
                    // future: predict_getData(),
                    builder: (context, snapshot) {
                      if (!snapshot.hasData) {
                        return Container();
                      }
                      // print(base64);
                      else {
                        print("base64 after pred: ${base64.substring(0, 200)}");

                        var uridata = Uri.parse(base64).data;
                        if (uridata != null) {
                          MemoryImage memoryImage =
                              MemoryImage(uridata.contentAsBytes());
                          return Container(
                            width: 100,
                            height: 275.0,
                            decoration: BoxDecoration(
                                image: DecorationImage(
                              image: memoryImage,
                              fit: BoxFit.cover,
                            )),
                          );
                        }
                        return Container(
                          child: Text("CANT GET PLOT IMAGE"),
                        );
                      }
                    },
                  )),
            ],
            // ),
          ),
        ),
      ),
      floatingActionButton: FloatingActionButton(
        onPressed: () {
          setState(() {
            b_predRun = !b_predRun;
            pose_to_predict = 11;
            testUser = 99;
            idxFrom = 0;
          });
        },
        tooltip: 'Said Hello',
        child: const Icon(Icons.start_outlined),
      ),
    );
  }
}

// ./starter-kit/prepare-sources.sh --proto ./protos/services.proto --flutterDir ./app --pythonDir ./server

class User_pose_selector extends StatefulWidget {
  const User_pose_selector({super.key});

  @override
  State<User_pose_selector> createState() => _User_pose_selectorState();
}

int testUser = 99;
var pose_to_predict = 0;
const List<String> UserList = <String>['One', 'Two'];
const List<String> PoseList = <String>['0', '1', '2', '3', '4', '5'];

class _User_pose_selectorState extends State<User_pose_selector> {
  String dropdownValue = UserList.first;

  @override
  Widget build(BuildContext context) {
    return Row(
      mainAxisAlignment: MainAxisAlignment.center,
      children: [
        Column(
          children: [
            Text("User"),
            DropdownMenu<String>(
              initialSelection: UserList.first,
              onSelected: (String? value) {
                // This is called when the user selects an item.
                setState(() {
                  dropdownValue = value!;
                  if (dropdownValue == "One") {
                    testUser = 0;
                  } else if (dropdownValue == "Two") {
                    testUser = 1;
                  } else {
                    testUser = 99;
                  }
                });
              },
              dropdownMenuEntries:
                  UserList.map<DropdownMenuEntry<String>>((String value) {
                return DropdownMenuEntry<String>(value: value, label: value);
              }).toList(),
            )
          ],
        ),
        Column(
          children: [
            Text("Pose"),
            DropdownMenu<String>(
              initialSelection: PoseList.first,
              onSelected: (String? value) {
                // This is called when the user selects an item.
                setState(() {
                  dropdownValue = value!;
                  pose_to_predict = int.parse(dropdownValue);
                  b_predRun = true;
                });
              },
              dropdownMenuEntries:
                  PoseList.map<DropdownMenuEntry<String>>((String value) {
                return DropdownMenuEntry<String>(value: value, label: value);
              }).toList(),
            )
          ],
        ),
      ],
    );
  }
}
