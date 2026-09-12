import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';

import 'features/fixtures/presentation/screens/fixtures_screen.dart';

final routerProvider = Provider<GoRouter>((ref) {
  return GoRouter(
    initialLocation: '/fixtures',
    routes: [
      GoRoute(
        path: '/fixtures',
        builder: (context, state) => const FixturesScreen(),
      ),
      GoRoute(
        path: '/',
        builder: (context, state) => const FixturesScreen(),
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
      title: 'PronosPro',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(
          seedColor: const Color(0xFF1D4ED8),
          brightness: Brightness.light,
        ),
        useMaterial3: true,
      ),
      routerConfig: router,
    );
  }
}