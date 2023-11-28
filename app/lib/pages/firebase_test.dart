import 'package:flutter/material.dart';

class FB_test extends StatefulWidget {
  const FB_test({super.key});

  @override
  State<FB_test> createState() => _TestState();
}

class _TestState extends State<FB_test> {
  var counter = 0;
  String greeting = '';
  void _increment() {
    setState(() {
      counter++;
    });
  }

  void _updateGreeting(String str_data) {
    setState(() {
      greeting = str_data;
    });
  }

  @override
  Widget build(BuildContext context) {
    TextEditingController testController = TextEditingController();
    void greateUser() {
      String data = testController.text;
      _updateGreeting("welcome, $data");
    }

    return Scaffold(
      appBar: AppBar(
        backgroundColor: Colors.amber,
        centerTitle: true,
        title: const Text(
          'FIRE BASE',
          style: TextStyle(fontWeight: FontWeight.bold, fontSize: 48),
        ),
      ),
      body: Text('FIRE BASE'),
    );
  }
}
