import 'package:flutter/material.dart';
import '../services/audio_service.dart';

class VolumeSlider extends StatelessWidget {
  final AudioService audioService;

  const VolumeSlider({required this.audioService, super.key});

  @override
  Widget build(BuildContext context) {
    return StreamBuilder<double>(
      stream: audioService.player.volumeStream,
      builder: (context, snapshot) {
        return Row(
          children: [
            const Icon(Icons.volume_up),
            Expanded(
              child: Slider(
                min: 0,
                max: 1,
                value: snapshot.data ?? 1,
                onChanged: (value) async {
                  await audioService.player.setVolume(value);
                },
              ),
            ),
          ],
        );
      },
    );
  }
}
