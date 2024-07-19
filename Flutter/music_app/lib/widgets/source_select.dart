import 'package:flutter/material.dart';
import '../services/audio_service.dart';
import '../models/track.dart';

class SourceSelect extends StatelessWidget {
  final AudioService audioService;

  const SourceSelect({required this.audioService, super.key});

  @override
  Widget build(BuildContext context) {
    return Column(
      children: audioService.playlist.map((track) {
        return ListTile(
          title: Text(track.title),
          subtitle: Text(track.artist),
          onTap: () async {
            await audioService.loadAudioSource(track);
            audioService.player.play();
          },
        );
      }).toList(),
    );
  }
}
