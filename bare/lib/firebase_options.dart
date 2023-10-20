// File generated by FlutterFire CLI.
// ignore_for_file: lines_longer_than_80_chars, avoid_classes_with_only_static_members
import 'package:firebase_core/firebase_core.dart' show FirebaseOptions;
import 'package:flutter/foundation.dart'
    show defaultTargetPlatform, kIsWeb, TargetPlatform;

/// Default [FirebaseOptions] for use with your Firebase apps.
///
/// Example:
/// ```dart
/// import 'firebase_options.dart';
/// // ...
/// await Firebase.initializeApp(
///   options: DefaultFirebaseOptions.currentPlatform,
/// );
/// ```
class DefaultFirebaseOptions {
  static FirebaseOptions get currentPlatform {
    if (kIsWeb) {
      return web;
    }
    switch (defaultTargetPlatform) {
      case TargetPlatform.android:
        return android;
      case TargetPlatform.iOS:
        return ios;
      case TargetPlatform.macOS:
        return macos;
      case TargetPlatform.windows:
        throw UnsupportedError(
          'DefaultFirebaseOptions have not been configured for windows - '
          'you can reconfigure this by running the FlutterFire CLI again.',
        );
      case TargetPlatform.linux:
        throw UnsupportedError(
          'DefaultFirebaseOptions have not been configured for linux - '
          'you can reconfigure this by running the FlutterFire CLI again.',
        );
      default:
        throw UnsupportedError(
          'DefaultFirebaseOptions are not supported for this platform.',
        );
    }
  }

  static const FirebaseOptions web = FirebaseOptions(
    apiKey: 'AIzaSyCtcpxvRvi9AOheQjdNU355brGpEp05eVo',
    appId: '1:303682390254:web:acc7e96ea587e3a2c80bff',
    messagingSenderId: '303682390254',
    projectId: 'arm-cyberware',
    authDomain: 'arm-cyberware.firebaseapp.com',
    storageBucket: 'arm-cyberware.appspot.com',
    measurementId: 'G-KZVNR9JJBJ',
  );

  static const FirebaseOptions android = FirebaseOptions(
    apiKey: 'AIzaSyBV2Hj7WQPCaIxU_xdDhOnZ0UG8TNMwsQc',
    appId: '1:303682390254:android:400dcd59756b1527c80bff',
    messagingSenderId: '303682390254',
    projectId: 'arm-cyberware',
    storageBucket: 'arm-cyberware.appspot.com',
  );

  static const FirebaseOptions ios = FirebaseOptions(
    apiKey: 'AIzaSyDgxKEm1Cz9G3M_Yyk4_aZoZvsEu2jBA2A',
    appId: '1:303682390254:ios:72d30b22ad85d922c80bff',
    messagingSenderId: '303682390254',
    projectId: 'arm-cyberware',
    storageBucket: 'arm-cyberware.appspot.com',
    iosBundleId: 'com.example.bare',
  );

  static const FirebaseOptions macos = FirebaseOptions(
    apiKey: 'AIzaSyDgxKEm1Cz9G3M_Yyk4_aZoZvsEu2jBA2A',
    appId: '1:303682390254:ios:381de7e6a9fc37d2c80bff',
    messagingSenderId: '303682390254',
    projectId: 'arm-cyberware',
    storageBucket: 'arm-cyberware.appspot.com',
    iosBundleId: 'com.example.bare.RunnerTests',
  );
}
