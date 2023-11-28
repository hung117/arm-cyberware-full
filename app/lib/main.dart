// Copyright 2017 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

import 'dart:async';

import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import './pages/home.dart';
import './pages/about.dart';
import './pages/page2.dart';
import './pages/test.dart';
import './pages/todo.dart';
import './pages/loading.dart';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();

  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Firebase Analytics Demo',
      theme: ThemeData(
        primarySwatch: Colors.blue,
      ),
      // navigatorObservers: <NavigatorObserver>[observer],

      // home: MyHomePage(
      //   title: 'Firebase Analytics Demo',
      //   analytics: analytics,
      //   observer: observer,
      // ),
      initialRoute: '/home',
      routes: {
        '/': (context) => Loading_screen(),
        '/home': (context) => HomePage(),
        '/page2': (context) => Page2(),
        '/test': (context) => Test(),
        '/todo': (context) => Todo(),
        '/about': (context) => About(),
      },
    );
  }
}
