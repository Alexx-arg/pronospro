import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';

import 'features/fixtures/presentation/screens/fixtures_screen.dart';
import 'features/prediction/presentation/screens/prediction_screen.dart';

final routerProvider = Provider<GoRouter>((ref) {
  return GoRouter(
    initialLocation: '/fixtures',
    routes: [
      GoRoute(
        path: '/fixtures',
        builder: (context, state) => const FixturesScreen(),
      ),
      GoRoute(
        path: '/prediction/:fixtureId',
        builder: (context, state) {
          final fixtureId = int.tryParse(state.pathParameters['fixtureId'] ?? '');
          return PredictionScreen(fixtureId: fixtureId);
        },
      ),
    ],
  );
});

class MyApp extends ConsumerWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final router = ref.watch(routerProvider);
    return MaterialApp.router(
      title: 'Football Prediction',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.blue),
        useMaterial3: true,
      ),
      routerConfig: router,
    );
  }
}