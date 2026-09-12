import 'package:flutter/material.dart';
import 'package:intl/intl.dart';

/// Horizontal date selector: Ayer · Hoy · Mañana · +5 días.
class DateSelectorBar extends StatelessWidget {
  const DateSelectorBar({
    super.key,
    required this.offset,
    required this.onSelect,
  });

  /// Day offset relative to today: -1 (ayer), 0 (hoy), 1..6 (siguientes).
  final int offset;
  final void Function(int) onSelect;

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    final now = DateTime.now();

    return SizedBox(
      height: 72,
      child: ListView.builder(
        scrollDirection: Axis.horizontal,
        padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 6),
        itemCount: 8, // -1 .. +6
        itemBuilder: (context, index) {
          final off = index - 1;
          final day = now.add(Duration(days: off));
          final selected = off == offset;
          return _DateChip(
            day: day,
            off: off,
            selected: selected,
            color: scheme.primary,
            onTap: () => onSelect(off),
          );
        },
      ),
    );
  }
}

class _DateChip extends StatelessWidget {
  const _DateChip({
    required this.day,
    required this.off,
    required this.selected,
    required this.color,
    required this.onTap,
  });

  final DateTime day;
  final int off;
  final bool selected;
  final Color color;
  final VoidCallback onTap;

  String get _label {
    switch (off) {
      case -1:
        return 'Ayer';
      case 0:
        return 'Hoy';
      case 1:
        return 'Mañana';
      default:
        return DateFormat('EEEE', 'es').format(day);
    }
  }

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    return GestureDetector(
      onTap: onTap,
      child: AnimatedContainer(
        duration: const Duration(milliseconds: 160),
        margin: const EdgeInsets.symmetric(horizontal: 4),
        padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 8),
        decoration: BoxDecoration(
          color: selected ? color : scheme.surfaceContainerHighest,
          borderRadius: BorderRadius.circular(12),
        ),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Text(
              _label,
              style: Theme.of(context).textTheme.labelLarge?.copyWith(
                    color: selected ? scheme.onPrimary : scheme.onSurfaceVariant,
                    fontWeight: selected ? FontWeight.w700 : FontWeight.w500,
                  ),
            ),
            const SizedBox(height: 2),
            Text(
              DateFormat('d/M', 'es').format(day),
              style: Theme.of(context).textTheme.labelSmall?.copyWith(
                    color: selected ? scheme.onPrimary : scheme.onSurfaceVariant,
                  ),
            ),
          ],
        ),
      ),
    );
  }
}