/// Fixture model matching backend GET /api/v1/fixtures/upcoming response.
/// Includes nested competition, season, teams, metrics, and prediction.

class TeamInfo {
  const TeamInfo({
    required this.id,
    required this.externalId,
    required this.name,
    this.shortName,
    this.code,
    this.country,
    this.logo,
    this.venue,
  });

  final int id;
  final int externalId;
  final String name;
  final String? shortName;
  final String? code;
  final String? country;
  final String? logo;
  final String? venue;

  factory TeamInfo.fromJson(Map<String, dynamic> json) {
    return TeamInfo(
      id: json['id'] as int,
      externalId: json['external_id'] as int,
      name: json['name'] as String,
      shortName: json['short_name'] as String?,
      code: json['code'] as String?,
      country: json['country'] as String?,
      logo: json['logo'] as String?,
      venue: json['venue'] as String?,
    );
  }

  Map<String, dynamic> toJson() {
    return {
      'id': id,
      'external_id': externalId,
      'name': name,
      'short_name': shortName,
      'code': code,
      'country': country,
      'logo': logo,
      'venue': venue,
    };
  }
}

class CompetitionInfo {
  const CompetitionInfo({
    required this.id,
    required this.externalId,
    required this.name,
    this.logo,
  });

  final int id;
  final int externalId;
  final String name;
  final String? logo;

  factory CompetitionInfo.fromJson(Map<String, dynamic> json) {
    return CompetitionInfo(
      id: json['id'] as int,
      externalId: json['external_id'] as int,
      name: json['name'] as String,
      logo: json['logo'] as String?,
    );
  }

  Map<String, dynamic> toJson() {
    return {
      'id': id,
      'external_id': externalId,
      'name': name,
      'logo': logo,
    };
  }
}

class SeasonInfo {
  const SeasonInfo({
    required this.id,
    required this.externalId,
    required this.year,
  });

  final int id;
  final int externalId;
  final int year;

  factory SeasonInfo.fromJson(Map<String, dynamic> json) {
    return SeasonInfo(
      id: json['id'] as int,
      externalId: json['external_id'] as int,
      year: json['year'] as int,
    );
  }

  Map<String, dynamic> toJson() {
    return {
      'id': id,
      'external_id': externalId,
      'year': year,
    };
  }
}

class MatchMetrics {
  const MatchMetrics({
    this.homeForm,
    this.awayForm,
    this.homeXg,
    this.homeXga,
    this.awayXg,
    this.awayXga,
    this.homeCornersAvg,
    this.awayCornersAvg,
    this.homeYellowCardsAvg,
    this.awayYellowCardsAvg,
    this.homeRedCardsAvg,
    this.awayRedCardsAvg,
    this.homePossessionAvg,
    this.awayPossessionAvg,
  });

  final String? homeForm;
  final String? awayForm;
  final double? homeXg;
  final double? homeXga;
  final double? awayXg;
  final double? awayXga;
  final double? homeCornersAvg;
  final double? awayCornersAvg;
  final double? homeYellowCardsAvg;
  final double? awayYellowCardsAvg;
  final double? homeRedCardsAvg;
  final double? awayRedCardsAvg;
  final double? homePossessionAvg;
  final double? awayPossessionAvg;

  factory MatchMetrics.fromJson(Map<String, dynamic> json) {
    return MatchMetrics(
      homeForm: json['home_form'] as String?,
      awayForm: json['away_form'] as String?,
      homeXg: (json['home_xg'] as num?)?.toDouble(),
      homeXga: (json['home_xga'] as num?)?.toDouble(),
      awayXg: (json['away_xg'] as num?)?.toDouble(),
      awayXga: (json['away_xga'] as num?)?.toDouble(),
      homeCornersAvg: (json['home_corners_avg'] as num?)?.toDouble(),
      awayCornersAvg: (json['away_corners_avg'] as num?)?.toDouble(),
      homeYellowCardsAvg: (json['home_yellow_cards_avg'] as num?)?.toDouble(),
      awayYellowCardsAvg: (json['away_yellow_cards_avg'] as num?)?.toDouble(),
      homeRedCardsAvg: (json['home_red_cards_avg'] as num?)?.toDouble(),
      awayRedCardsAvg: (json['away_red_cards_avg'] as num?)?.toDouble(),
      homePossessionAvg: (json['home_possession_avg'] as num?)?.toDouble(),
      awayPossessionAvg: (json['away_possession_avg'] as num?)?.toDouble(),
    );
  }

  Map<String, dynamic> toJson() {
    return {
      'home_form': homeForm,
      'away_form': awayForm,
      'home_xg': homeXg,
      'home_xga': homeXga,
      'away_xg': awayXg,
      'away_xga': awayXga,
      'home_corners_avg': homeCornersAvg,
      'away_corners_avg': awayCornersAvg,
      'home_yellow_cards_avg': homeYellowCardsAvg,
      'away_yellow_cards_avg': awayYellowCardsAvg,
      'home_red_cards_avg': homeRedCardsAvg,
      'away_red_cards_avg': awayRedCardsAvg,
      'home_possession_avg': homePossessionAvg,
      'away_possession_avg': awayPossessionAvg,
    };
  }
}

class PredictionInfo {
  const PredictionInfo({
    required this.id,
    required this.modelVersion,
    required this.homeProbability,
    required this.drawProbability,
    required this.awayProbability,
    this.expectedHomeGoals,
    this.expectedAwayGoals,
    this.confidence,
    required this.createdAt,
  });

  final int id;
  final String modelVersion;
  final double homeProbability;
  final double drawProbability;
  final double awayProbability;
  final double? expectedHomeGoals;
  final double? expectedAwayGoals;
  final double? confidence;
  final DateTime createdAt;

  factory PredictionInfo.fromJson(Map<String, dynamic> json) {
    return PredictionInfo(
      id: json['id'] as int,
      modelVersion: json['model_version'] as String,
      homeProbability: (json['home_probability'] as num).toDouble(),
      drawProbability: (json['draw_probability'] as num).toDouble(),
      awayProbability: (json['away_probability'] as num).toDouble(),
      expectedHomeGoals: (json['expected_home_goals'] as num?)?.toDouble(),
      expectedAwayGoals: (json['expected_away_goals'] as num?)?.toDouble(),
      confidence: (json['confidence'] as num?)?.toDouble(),
      createdAt: DateTime.parse(json['created_at'] as String),
    );
  }

  Map<String, dynamic> toJson() {
    return {
      'id': id,
      'model_version': modelVersion,
      'home_probability': homeProbability,
      'draw_probability': drawProbability,
      'away_probability': awayProbability,
      'expected_home_goals': expectedHomeGoals,
      'expected_away_goals': expectedAwayGoals,
      'confidence': confidence,
      'created_at': createdAt.toIso8601String(),
    };
  }
}

class FixtureResponse {
  const FixtureResponse({
    required this.id,
    required this.externalId,
    required this.competition,
    required this.season,
    required this.homeTeam,
    required this.awayTeam,
    required this.kickoffTime,
    required this.status,
    this.statusShort,
    this.venue,
    this.homeGoals,
    this.awayGoals,
    this.metrics,
    this.prediction,
  });

  final int id;
  final int externalId;
  final CompetitionInfo competition;
  final SeasonInfo season;
  final TeamInfo homeTeam;
  final TeamInfo awayTeam;
  final DateTime kickoffTime;
  final String status;
  final String? statusShort;
  final String? venue;
  final int? homeGoals;
  final int? awayGoals;
  final MatchMetrics? metrics;
  final PredictionInfo? prediction;

  factory FixtureResponse.fromJson(Map<String, dynamic> json) {
    return FixtureResponse(
      id: json['id'] as int,
      externalId: json['external_id'] as int,
      competition: CompetitionInfo.fromJson(json['competition'] as Map<String, dynamic>),
      season: SeasonInfo.fromJson(json['season'] as Map<String, dynamic>),
      homeTeam: TeamInfo.fromJson(json['home_team'] as Map<String, dynamic>),
      awayTeam: TeamInfo.fromJson(json['away_team'] as Map<String, dynamic>),
      kickoffTime: DateTime.parse(json['kickoff_time'] as String),
      status: json['status'] as String,
      statusShort: json['status_short'] as String?,
      venue: json['venue'] as String?,
      homeGoals: json['home_goals'] as int?,
      awayGoals: json['away_goals'] as int?,
      metrics: json['metrics'] != null ? MatchMetrics.fromJson(json['metrics'] as Map<String, dynamic>) : null,
      prediction: json['prediction'] != null ? PredictionInfo.fromJson(json['prediction'] as Map<String, dynamic>) : null,
    );
  }

  Map<String, dynamic> toJson() {
    return {
      'id': id,
      'external_id': externalId,
      'competition': competition.toJson(),
      'season': season.toJson(),
      'home_team': homeTeam.toJson(),
      'away_team': awayTeam.toJson(),
      'kickoff_time': kickoffTime.toIso8601String(),
      'status': status,
      'status_short': statusShort,
      'venue': venue,
      'home_goals': homeGoals,
      'away_goals': awayGoals,
      'metrics': metrics?.toJson(),
      'prediction': prediction?.toJson(),
    };
  }
}

class PaginatedFixtures {
  const PaginatedFixtures({
    required this.items,
    required this.limit,
    required this.offset,
    required this.total,
  });

  final List<FixtureResponse> items;
  final int limit;
  final int offset;
  final int total;

  factory PaginatedFixtures.fromJson(Map<String, dynamic> json) {
    return PaginatedFixtures(
      items: (json['items'] as List)
          .map((e) => FixtureResponse.fromJson(e as Map<String, dynamic>))
          .toList(),
      limit: json['limit'] as int,
      offset: json['offset'] as int,
      total: json['total'] as int,
    );
  }

  Map<String, dynamic> toJson() {
    return {
      'items': items.map((e) => e.toJson()).toList(),
      'limit': limit,
      'offset': offset,
      'total': total,
    };
  }
}