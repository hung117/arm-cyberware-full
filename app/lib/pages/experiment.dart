import 'package:flutter/material.dart';
import 'dart:math';
import 'dart:ui';
import 'package:flutter/material.dart';
import 'package:app/grpc_generated/client.dart';
import 'package:app/grpc_generated/init_py.dart';
// import 'package:app/grpc_generated/service.pbgrpc.dart';
import '../grpc_generated/services.pbgrpc.dart';
import '../custom_widgets/pythonLoaderMonitor.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:image_picker/image_picker.dart';
import 'dart:async';
import 'dart:io';
import 'package:path_provider/path_provider.dart';

Future<void> pyInitResult = Future(() => null);
Future<void> pyImportResult = Future(() => null);

class Experiment extends StatefulWidget {
  const Experiment({Key? key}) : super(key: key);

  @override
  State<Experiment> createState() => ExperimentState();
}

// class ExperimentState extends State<Experiment> with WidgetsBindingObserver {
class ExperimentState extends State<Experiment> {
  List<int> randomIntegers =
      List.generate(40, (index) => Random().nextInt(100));
  bool b_predRun = false;
  int iPose = 0;
  late var bytes;
  late Image _image;

  void getImageByte() async {
    bytes = await new File('assets/signal.png').readAsBytes();
  }

  @override
  void initState() {
    WidgetsFlutterBinding.ensureInitialized();
    pyInitResult = initPy();
    super.initState();
    // count();
    // getImageByte();
  }

  void update() {
    setState(() {
      iPose += 1;
      if (iPose > 5) {
        iPose = 0;
      }
    });
  }

  String path_str = 'assets/signal.png';
  int sample_idx = 0;
  Future<void> count() async {
    // Make an HTTP request to fetch some data
    setState(() {
      iPose += 1;
      if (iPose > 5) {
        iPose = 0;
      }
      count();
    });
    print(iPose);
    Future.delayed(const Duration(milliseconds: 10));
    // Return the response body as a string
  }

  Future getImage() async {
    // using your method of getting an image
    Image newImage = Image.file(
      File('/storage/emulated/0/Download/test.jpg'),
    );
    setState(() {
      _image = newImage;
    });
  }

  @override
  Widget build(BuildContext context) {
    WidgetsBinding.instance.addPostFrameCallback((_) => setState(() {
          if (b_predRun) {
            // iPose += 1;
            // iPose = (iPose > 5) ? 0 : iPose;
            EMGClassifierServiceClient(getClientChannel())
                .classify_Signal(PlaceHolderMsg())
                // .then((p0) => iPose = p0.signal);
                .then((p0) => {sample_idx++, iPose = p0.signal});
          }
        }));
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
              Text("IPOSE: ${iPose}"),
              SizedBox(
                height: 50,
                child: PythonLoaderMonitor(),
              ),
              const SizedBox(height: 16),
              ElevatedButton(
                onPressed: () {
                  setState(() {
                    iPose += 1;
                    iPose = (iPose > 5) ? 0 : iPose;
                  });
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
                        getImage();
                        // EMGClassifierServiceClient(getClientChannel())
                        //     .classify_Signal(PlaceHolderMsg())
                        //     .then((p0) => iPose = p0.signal);
                        // update();
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
                  child: Image(image: AssetImage('assets/signal.png'))),
              // child: Image.memory(Uint8List.fromList(bytes))),
              // child: Image.memory(bytes)),
              // child: Image.memory(),
              // child: Image(image: AssetImage('assets/signal.png'))),
              // image: FileImage(File('assets/signal.png')))),
              // image: FileImage(File('assets/signal.png')))),
            ],
            // ),
          ),
        ),
      ),
      floatingActionButton: FloatingActionButton(
        onPressed: () => {},
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
