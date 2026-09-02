import '../../data/models/fixture_response.dart';

sealed class FixturesState {
  const FixturesState();
}

class FixturesInitial extends FixturesState {
  const FixturesInitial();
}

class FixturesLoading extends FixturesState {
  const FixturesLoading();
}

class FixturesSuccess extends FixturesState {
  const FixturesSuccess(this.fixtures, this.hasReachedMax);
  final PaginatedFixtures fixtures;
  final bool hasReachedMax;
}

class FixturesError extends FixturesState {
  const FixturesError(this.message);
  final String message;
}

class FixtureDetailState {
  const FixtureDetailState();
}

class FixtureDetailInitial extends FixtureDetailState {
  const FixtureDetailInitial();
}

class FixtureDetailLoading extends FixtureDetailState {
  const FixtureDetailLoading();
}

class FixtureDetailSuccess extends FixtureDetailState {
  const FixtureDetailSuccess(this.fixture);
  final FixtureResponse fixture;
}

class FixtureDetailError extends FixtureDetailState {
  const FixtureDetailError(this.message);
  final String message;
}