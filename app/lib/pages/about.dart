import 'package:flutter/material.dart';

class About extends StatelessWidget {
  const About({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        backgroundColor: Colors.blueGrey,
        centerTitle: true,
        title: const Text(
          'ABOUT',
          style: TextStyle(fontWeight: FontWeight.bold, fontSize: 48),
        ),
      ),
      body: Container(
        color: Colors.blueAccent,
      ),
      bottomNavigationBar: BottomNavigationBar(items: [
        BottomNavigationBarItem(
            icon: Icon(Icons.broadcast_on_home), label: 'placeholder'),
        BottomNavigationBarItem(
            icon: Icon(Icons.cyclone), label: 'placeholder'),
      ]),
    );
  }
}
