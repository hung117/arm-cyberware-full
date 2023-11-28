import 'package:flutter/material.dart';
import 'package:flutter_spinkit/flutter_spinkit.dart';

class Loading_screen extends StatefulWidget {
  const Loading_screen({super.key});

  @override
  State<Loading_screen> createState() => _WeatherHomePageState();
}

class _WeatherHomePageState extends State<Loading_screen> {
  Map data = {'location': 'London', 'url': 'Europe/London'};

  @override
  void initState() {
    print('INIT MAP DATA: ${data}');
    // TODO: implement initState
    super.initState();

    print('data from pick loc: ${data}');
  }

  @override
  Widget build(BuildContext context) {
    data = ModalRoute.of(context)!.settings.arguments as Map;
    return Scaffold(
      backgroundColor: Colors.grey[900],
      body: Center(
          child: SpinKitDancingSquare(
        // child: SpinKitRotatingCircle(
        color: Colors.white,
        size: 400.0,
      )),
    );
  }
}
