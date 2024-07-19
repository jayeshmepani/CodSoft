import 'package:flutter/material.dart';
import '../services/audio_service.dart';

class ControlButtons extends StatelessWidget {
  final AudioService audioService;

  const ControlButtons({required this.audioService, super.key});

  @override
  Widget build(BuildContext context) {
    return Column(
      mainAxisSize: MainAxisSize.min,
      children: [
        StreamBuilder<double>(
          stream: audioService.player.speedStream,
          builder: (context, snapshot) {
            return Row(
              children: [
                const Icon(Icons.speed),
                Expanded(
                  child: TextField(
                    keyboardType:
                        TextInputType.numberWithOptions(decimal: true),
                    decoration: InputDecoration(
                      hintText: (snapshot.data ?? 1.0).toStringAsFixed(2),
                      labelText: 'Speed (0.2 - 5.0)',
                    ),
                    onSubmitted: (value) async {
                      double? speed = double.tryParse(value);
                      if (speed != null && speed >= 0.2 && speed <= 5.0) {
                        await audioService.player.setSpeed(speed);
                      } else {
                        ScaffoldMessenger.of(context).showSnackBar(
                          SnackBar(
                              content:
                                  Text('Enter a value between 0.2 and 5.0')),
                        );
                      }
                    },
                  ),
                ),
              ],
            );
          },
        ),
        StreamBuilder<double>(
          stream: audioService.player.volumeStream,
          builder: (context, snapshot) {
            return Row(
              children: [
                const Icon(Icons.volume_up),
                Expanded(
                  child: Slider(
                    min: 0.0,
                    max: 1.0,
                    value: snapshot.data ?? 1.0,
                    // Omitting divisions for a smooth slider
                    onChanged: (value) async {
                      await audioService.player.setVolume(value);
                    },
                  ),
                ),
              ],
            );
          },
        ),
      ],
    );
  }
}
