import 'package:flutter/material.dart';
import '../pages/experiment.dart';

class PythonLoaderMonitor extends StatelessWidget {
  const PythonLoaderMonitor({
    super.key,
  });

  @override
  Widget build(BuildContext context) {
    return FutureBuilder<void>(
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
          print(
              "ERRRORRR UHH OH \n error: ${snapshot.error}\n stackTrace: ${snapshot.stackTrace} \n ERRRORRR REPORT END");

          return Text(
            'Error: ${snapshot.error}, and ${snapshot.stackTrace}',
            style: TextStyle(color: Colors.red, fontSize: 20),
          );
        } else {
          // When future completes, display a message saying that Python has been loaded
          // Set the text color of the Text widget to green
          return const Text(
            'Python is ready',
            style: TextStyle(color: Colors.green, fontSize: 30),
          );
        }
      },
    );
  }
}

class NNLoadMonitor extends StatelessWidget {
  const NNLoadMonitor({
    super.key,
  });
  // import of NN code

  @override
  Widget build(BuildContext context) {
    return FutureBuilder<void>(
      future: pyImportResult,
      builder: (context, snapshot) {
        if (snapshot.connectionState == ConnectionState.waiting) {
          return const Stack(
            children: [
              SizedBox(height: 4, child: LinearProgressIndicator()),
              Positioned.fill(
                child: Center(
                  child: Text(
                    'Imporing...',
                  ),
                ),
              ),
            ],
          ); // Add FutureBuilder that awaits pyInitResult
        } else if (snapshot.hasError) {
          // If error is returned by the future, display an error message
          print(
              "ERRRORRR UHH OH \n importing error: ${snapshot.error}\n stackTrace: ${snapshot.stackTrace} \n ERRRORRR REPORT END");

          return Text(
            'Error: ${snapshot.error}, and ${snapshot.stackTrace}',
            style: TextStyle(color: Colors.red, fontSize: 20),
          );
        } else {
          // When future completes, display a message saying that Python has been loaded
          // Set the text color of the Text widget to green
          return const Text(
            'Python is ready',
            style: TextStyle(color: Colors.green, fontSize: 30),
          );
        }
      },
    );
  }
}
