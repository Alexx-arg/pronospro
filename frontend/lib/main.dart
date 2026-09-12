import 'package:flutter/material.dart';
import 'package:flutter_dotenv/flutter_dotenv.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:intl/date_symbol_data_local.dart';

import 'app.dart';

/// Bootstrap limpio — D13.2
Future<void> bootstrap() async {
  WidgetsFlutterBinding.ensureInitialized();
  // Necesario para DateFormat con locale 'es' (DateSelectorBar, fixture cards).
  await initializeDateFormatting('es', null);
  await _loadEnv();
}

Future<void> _loadEnv() async {
  // D13.1: tolerante a ausencia de .env (CI/CD, --dart-define, tests)
  try {
    await dotenv.load(fileName: ".env");
  } catch (_) {
    // .env no existe o no es legible — cae a defaults / String.fromEnvironment
  }
}

Future<void> main() async {
  await bootstrap();
  runApp(const ProviderScope(child: MyApp()));
}
