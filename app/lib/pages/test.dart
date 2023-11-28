import 'package:flutter/material.dart';
import 'dart:math';
import 'dart:ui';
import 'package:flutter/material.dart';
import 'package:app/grpc_generated/client.dart';
import 'package:app/grpc_generated/init_py.dart';
// import 'package:app/grpc_generated/service.pbgrpc.dart';
import '../grpc_generated/services.pbgrpc.dart';

Future<void> pyInitResult = Future(() => null);

class Test extends StatefulWidget {
  const Test({Key? key}) : super(key: key);

  @override
  TestState createState() => TestState();
}

class TestState extends State<Test> with WidgetsBindingObserver {
  List<int> randomIntegers =
      List.generate(40, (index) => Random().nextInt(100));
  @override
  Future<AppExitResponse> didRequestAppExit() {
    shutdownPyIfAny();
    return super.didRequestAppExit();
  }

  @override
  void initState() {
    WidgetsFlutterBinding.ensureInitialized();
    pyInitResult = initPy();
    super.initState();
    WidgetsBinding.instance.addObserver(this);
  }

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: Scaffold(
        body: Container(
          padding: const EdgeInsets.all(20),
          alignment: Alignment.center,
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              SizedBox(
                height: 50,
                child: FutureBuilder<void>(
                  future: pyInitResult,
                  builder: (context, snapshot) {
                    if (snapshot.connectionState == ConnectionState.waiting) {
                      return const Stack(
                        children: [
                          SizedBox(height: 4, child: LinearProgressIndicator()),
                          Positioned.fill(
                            child: Center(
                              child: Text(
                                'Loading Python...',
                              ),
                            ),
                          ),
                        ],
                      ); // Add FutureBuilder that awaits pyInitResult
                    } else if (snapshot.hasError) {
                      // If error is returned by the future, display an error message
                      print("ERRRORRR UHH OH");
                      print(snapshot.error);
                      print(snapshot.stackTrace);
                      print("ERRRORRR END");

                      return Text(
                        'Error: ${snapshot.error}, and ${snapshot.stackTrace}',
                        style: TextStyle(color: Colors.red, fontSize: 20),
                      );
                    } else {
                      // When future completes, display a message saying that Python has been loaded
                      // Set the text color of the Text widget to green
                      return const Text(
                        'Python has been loaded',
                        style: TextStyle(color: Colors.green, fontSize: 30),
                      );
                    }
                  },
                ),
              ),
              const SizedBox(height: 16),
              Text(
                randomIntegers.join(', '),
                textAlign: TextAlign.center,
              ),
              const SizedBox(height: 16),
              ElevatedButton(
                onPressed: () {
                  setState(() {
                    randomIntegers =
                        List.generate(40, (index) => Random().nextInt(100));
                  });
                },
                style: ElevatedButton.styleFrom(
                  minimumSize:
                      const Size(140, 36), // Set minimum width to 120px
                ),
                child: const Text('Regenerate List'),
              ),
              const SizedBox(height: 16),
              ElevatedButton(
                onPressed: () {
                  // setState(() => randomIntegers.sort());
                  NumberSortingServiceClient(getClientChannel())
                      .sortNumbers(NumberArray(numbers: randomIntegers))
                      .then(
                          (p0) => setState(() => randomIntegers = p0.numbers));
                },
                style: ElevatedButton.styleFrom(
                  minimumSize:
                      const Size(140, 36), // Set minimum width to 120px
                ),
                child: const Text('Sort'),
              ),
              MyCustomForm(),
            ],
          ),
        ),
      ),
    );
  }
}

class MyCustomForm extends StatefulWidget {
  const MyCustomForm({super.key});

  @override
  MyCustomFormState createState() {
    return MyCustomFormState();
  }
}

class MyCustomFormState extends State<MyCustomForm> {
  // Create a global key that uniquely identifies the Form widget
  // and allows validation of the form.
  //
  // Note: This is a GlobalKey<FormState>,
  // not a GlobalKey<MyCustomFormState>.
  final _formKey = GlobalKey<FormState>();
  final text_con_1 = TextEditingController();
  final text_con_2 = TextEditingController();
  @override
  void dispose() {
    // Clean up the controller when the widget is disposed.
    text_con_1.dispose();
    text_con_2.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    // Build a Form widget using the _formKey created above.
    return Form(
      key: _formKey,
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 16),
            child: TextFormField(
              onChanged: (text) {
                print('First text field: $text (${text.characters.length})');
              },
              // The validator receives the text that the user has entered.
              controller: text_con_1,
              validator: (value) {
                if (value == null || value.isEmpty) {
                  return 'Please enter some text';
                }
                return null;
              },
              decoration: const InputDecoration(
                border: UnderlineInputBorder(),
                labelText: 'reason',
              ),
            ),
          ),
          const SizedBox(
            height: 5,
          ),
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 16),
            child: TextFormField(
              // The validator receives the text that the user has entered.
              controller: text_con_2,
              validator: (value) {
                if (value == null || value.isEmpty) {
                  return 'Please enter some text';
                }
                return null;
              },
              decoration: const InputDecoration(
                border: OutlineInputBorder(),
                labelText: 'punchline',
              ),
            ),
          ),
          Padding(
            padding: const EdgeInsets.symmetric(vertical: 16),
            child: ElevatedButton(
              onPressed: () {
                UrMumJokeClient(getClientChannel())
                    .tellJoke(theJokeReq(
                        reason: text_con_1.text, punchline: text_con_2.text))
                    .then((p0) => {print(p0)});
                // UrMumJokeClient(getGrpcClientChannel(host, port, useHttps))

                showDialog(
                  context: context,
                  builder: (context) {
                    return AlertDialog(
                      // Retrieve the text that the user has entered by using the
                      // TextEditingController.
                      content: Text(
                          'ur mum is so ${text_con_1.text}, she ${text_con_2.text}'),
                    );
                  },
                );
                // Validate returns true if the form is valid, or false otherwise.
                if (_formKey.currentState!.validate()) {
                  // If the form is valid, display a snackbar. In the real world,
                  // you'd often call a server or save the information in a database.
                  ScaffoldMessenger.of(context).showSnackBar(
                    const SnackBar(content: Text('Processing Data')),
                  );
                }
              },
              child: const Text('Submit'),
            ),
          ),
        ],
      ),
    );
  }
}
// ./starter-kit/prepare-sources.sh --proto ./protos/services.proto --flutterDir ./app --pythonDir ./server