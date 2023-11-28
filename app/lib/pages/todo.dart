import 'package:flutter/material.dart';

class Todo extends StatelessWidget {
  Todo({super.key});
  List todo = [1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 1, 2, 3, 4, 5, 6, 7, 8, 9];
  Widget viewListContainer() {
    return ListView.builder(
      itemCount: todo.length,
      itemBuilder: (context, index) => Padding(
        padding: const EdgeInsets.all(10.0),
        child: Container(
          padding: const EdgeInsets.all(25.0),
          decoration: BoxDecoration(
              color: Colors.amber[todo[index] * 100],
              borderRadius: BorderRadius.circular(15)),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.start,
            children: [
              // C H E C K    B O X

              // T A S K    N A M E
              Text(todo[index].toString()),
              Text(todo[index].toString()),
              Text(todo[index].toString()),
            ],
          ),
        ),
      ),
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
    return Scaffold(
        backgroundColor: Colors.yellow[200],
        appBar: AppBar(
          backgroundColor: Colors.amber,
          centerTitle: true,
          title: const Text(
            'TO DO',
            style: TextStyle(
              fontWeight: FontWeight.normal,
              fontSize: 48,
            ),
          ),
        ),
        body: viewListContainer());
  }
}
