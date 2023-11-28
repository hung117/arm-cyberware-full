import 'package:flutter/material.dart';

class Page2 extends StatelessWidget {
  Page2({super.key});
  List<int> users = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];

  Widget viewListTile() {
    return ListView.builder(
      itemCount: users.length,
      itemBuilder: (context, index) => ListTile(
        title: Container(
            height: 300,
            color: Colors.amber[users[index] * 100],
            child: Text(users[index].toString())),
      ),
    );
  }

  Widget viewListContainer() {
    return ListView.builder(
      itemCount: users.length,
      itemBuilder: (context, index) => Container(
          height: 300,
          color: Colors.amber[users[index] * 100],
          child: Text(users[index].toString())),
    );
  }

  Widget viewGrid() {
    return GridView.builder(
        itemCount: 64,
        gridDelegate:
            SliverGridDelegateWithFixedCrossAxisCount(crossAxisCount: 4),
        itemBuilder: ((context, index) => Container(
              color: Colors.amber,
              margin: EdgeInsets.all(2),
            )));
  }

  @override
  Widget build(BuildContext context) {
    users = new List.from(users.reversed);
    print(users.length);
    return Scaffold(
        appBar: AppBar(
          backgroundColor: Colors.amber,
          centerTitle: true,
          title: const Text(
            'PAGE 2',
            style: TextStyle(fontWeight: FontWeight.bold, fontSize: 48),
          ),
          actions: [
            GestureDetector(
              onTap: () => {print("about pressed")},
              child: Container(
                margin: EdgeInsets.all(10),
                child: Icon(Icons.more),
                decoration: BoxDecoration(
                  borderRadius: BorderRadius.circular(10),
                  color: const Color(0xFFFFFF),
                ),
              ),
            )
          ],
        ),
        // body: viewListTile());
        body: viewListContainer());
    // body: viewGrid());
  }
}
