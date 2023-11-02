import 'package:flutter/material.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  // This widget is the root of your application.
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Flutter Demo',
      theme: ThemeData(
        colorScheme:
            ColorScheme.fromSeed(seedColor: Color.fromARGB(255, 31, 69, 63)),
        useMaterial3: true,
      ),
      home: const MyHomePage(title: 'Home Page'),
    );
  }
}

class MyHomePage extends StatefulWidget {
  const MyHomePage({super.key, required this.title});

  final String title;

  @override
  State<MyHomePage> createState() => _MyHomePageState();
}

class _MyHomePageState extends State<MyHomePage> {
  int page_idx = 0;
  int n_pages = 2;

  void _incrementCounter() {
    print(page_idx);
    setState(() {
      page_idx++;
      if (page_idx > n_pages) {
        page_idx = 0;
      }
      switch (page_idx) {
        case 0:
          // do something
          break;
        case 1:
          Navigator.push(
            context,
            MaterialPageRoute(builder: (context) => const FirstRoute()),
          );
          break;
        case 2:
          Navigator.push(
            context,
            MaterialPageRoute(builder: (context) => SecondRoute()),
          );
          break;
      }
    });
  }

  void _decrementCounter() {
    setState(() {
      page_idx--;
      if (page_idx <= 0) {
        page_idx = 0;
        Navigator.push(
          context,
          MaterialPageRoute(builder: (context) => SecondRoute()),
        );
      } else {
        Navigator.pop(context);
      }
    });
  }

  void switchPages() {}

  @override
  Widget build(BuildContext context) {
    return Scaffold(
        appBar: AppBar(
          backgroundColor: Theme.of(context).colorScheme.inversePrimary,
          title: Text(widget.title),
        ),
        body: Center(
          child: const Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: <Widget>[
              const Text(
                'place holder for home, this pages supposed to provide information',
              ),
            ],
          ),
        ),
        floatingActionButton: Row(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            FloatingActionButton(
              onPressed: _decrementCounter,
              tooltip: 'prev',
              child: const Icon(Icons.arrow_back_ios),
            ),
            Text(
              "                                       ",
              style: Theme.of(context).textTheme.headlineMedium,
            ),
            FloatingActionButton(
              onPressed: _incrementCounter,
              tooltip: 'next',
              child: const Icon(Icons.arrow_forward_ios),
            ),
          ],
        ));
  }
}

class FirstRoute extends StatelessWidget {
  const FirstRoute({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
        appBar: AppBar(
          backgroundColor: Theme.of(context).colorScheme.primary,
          title: const Text('Record new data'),
        ),
        body: Center(
          child: Column(mainAxisAlignment: MainAxisAlignment.center, children: [
            Padding(
              padding: const EdgeInsets.all(20.0),
              child: Text(
                  'Cosmetic for now, this page is for recording new data from your sensor'),
            ),
            ElevatedButton(
              onPressed: () {},
              child: Text('RECORD YOUR DATA'),
            )
          ]),
        ),
        floatingActionButton: Row(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            FloatingActionButton(
              onPressed: () {
                Navigator.pop(context);
              },
              tooltip: 'prev page',
              child: const Icon(Icons.arrow_back_ios),
            ),
            Text(
              "                                ",
              style: Theme.of(context).textTheme.headlineMedium,
            ),
            FloatingActionButton(
              onPressed: () {
                Navigator.push(
                  context,
                  MaterialPageRoute(builder: (context) => SecondRoute()),
                );
              },
              tooltip: 'next page',
              child: const Icon(Icons.arrow_forward_ios),
            ),
          ],
        ));
  }
}

class SecondRoute extends StatelessWidget {
  SecondRoute({super.key});
  String text =
      " asldfkj laskdfl asldkfj saldkjf asldkf j e diam luctus eu. Aliquam fermentum ac mauris vitae molestie. Etiam faucibus nunc a arcu blandit fermentum. Donec vitae elit turpi";
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        backgroundColor: Theme.of(context).colorScheme.inversePrimary,
        title: const Text('about or st'),
      ),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Text(text),
            ElevatedButton(
              onPressed: () {
                Navigator.pop(context);
              },
              child: const Text('return'),
            ),
          ],
        ),
      ),
    );
  }
}
