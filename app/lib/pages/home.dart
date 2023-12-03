import 'package:flutter/material.dart';
import 'package:app/pages/about.dart';
import 'package:app/pages/firebase_test.dart';
import 'package:app/pages/page2.dart';
import 'package:app/pages/test.dart';
import 'package:app/pages/todo.dart';
import './page2.dart';

class HomePage extends StatelessWidget {
  const HomePage({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color.fromARGB(255, 184, 225, 215),
      appBar: AppBar(
        backgroundColor: Colors.blueGrey,
        title: const Text('HOME',
            style: TextStyle(
                color: Colors.tealAccent,
                fontWeight: FontWeight.w500,
                fontSize: 48)),
        centerTitle: true,
      ),
      drawer: Drawer(
        backgroundColor: Colors.teal[100],
        child: Column(children: [
          const DrawerHeader(
              child: Icon(
            Icons.back_hand,
            size: 48,
          )),
          ListTile(
            onTap: () {
              Navigator.pop(context);
              // Navigator.push(
              // context, MaterialPageRoute(builder: (context) => About()));
              Navigator.pushNamed(context, '/about');
            },
            leading: Icon(Icons.info),
            title: const Text(
              'about',
              style: TextStyle(fontSize: 20, fontWeight: FontWeight.bold),
            ),
          ),
          ListTile(
            onTap: () {
              Navigator.pop(context);
              Navigator.push(
                  context, MaterialPageRoute(builder: (context) => Page2()));
            },
            leading: Icon(Icons.recommend_rounded),
            title: Text(
              'record new data',
              style: TextStyle(fontSize: 20, fontWeight: FontWeight.bold),
            ),
          ),
          ListTile(
            onTap: () {
              Navigator.pop(context);
              // Navigator.push(
              //     context, MaterialPageRoute(builder: (context) => FB_test()));
              Navigator.pushNamed(context, '/test');
            },
            leading: Icon(Icons.textsms),
            title: Text(
              'TEST',
              style: TextStyle(fontSize: 20, fontWeight: FontWeight.bold),
            ),
          ),
          ListTile(
            onTap: () {
              Navigator.pop(context);
              // Navigator.push(
              //     context, MaterialPageRoute(builder: (context) => FB_test()));
              Navigator.pushNamed(context, '/experiment');
            },
            leading: Icon(Icons.api),
            title: Text(
              'EXPERIMENT',
              style: TextStyle(fontSize: 20, fontWeight: FontWeight.bold),
            ),
          ),
          ListTile(
            onTap: () {
              Navigator.pop(context);
              // Navigator.push(
              //     context, MaterialPageRoute(builder: (context) => Todo()));
              Navigator.pushNamed(context, '/todo');
            },
            leading: Icon(Icons.list),
            title: Text(
              'TODO',
              style: TextStyle(fontSize: 20, fontWeight: FontWeight.bold),
            ),
          ),
        ]),
      ),
      body: Container(
        child: Row(
          children: [
            Column(
              children: [
                Expanded(
                  flex: 1,
                  child: Container(
                    width: 200,
                    color: Colors.teal,
                  ),
                ),
                Expanded(
                  flex: 2,
                  child: Container(
                    width: 200,
                    color: Colors.teal[400],
                  ),
                ),
                Expanded(
                  flex: 1,
                  child: Container(
                    width: 200,
                    color: Colors.teal[300],
                  ),
                ),
              ],
            ),
            Column(
              mainAxisAlignment: MainAxisAlignment.end,
              crossAxisAlignment: CrossAxisAlignment.center,
              children: [
                Expanded(
                    flex: 1,
                    child: GestureDetector(
                      onTap: () => (
                        // Navigator.push(
                        //     context,
                        //     MaterialPageRoute(
                        //       builder: (context) => About(),
                        //     )),
                        // Navigator.pop(context),
                        Navigator.pushNamed(context, '/about'),
                      ),
                      child: Container(
                          width: 280,
                          color: Colors.tealAccent[400],
                          child: const Center(
                              child: Text(
                            'About',
                            style: TextStyle(fontSize: 35),
                          ))),
                    )),
                Expanded(
                    flex: 2,
                    child: GestureDetector(
                      onTap: () => (
                          // Navigator.push(
                          //     context,
                          //     MaterialPageRoute(
                          //       builder: (context) => Page2(),
                          //     )),
                          Navigator.pushNamed(context, '/page2')),
                      child: Container(
                          width: 280,
                          color: Colors.tealAccent,
                          child: const Center(
                              child: Text(
                            'Page2',
                            style: TextStyle(fontSize: 35),
                          ))),
                    )),
                Expanded(
                  flex: 2,
                  child: Container(
                    width: 280,
                    color: Colors.tealAccent[400],
                  ),
                ),
              ],
            ),
          ],
        ),
      ),
      floatingActionButton: FloatingActionButton(
        onPressed: () => {},
        tooltip: 'Said Hello',
        child: const Icon(Icons.add),
      ),
    );
  }
}
