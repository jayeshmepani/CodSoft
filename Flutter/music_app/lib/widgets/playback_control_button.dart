import 'package:flutter/material.dart';
import 'package:just_audio/just_audio.dart';
import '../services/audio_service.dart';

class PlaybackControlButton extends StatelessWidget {
  final AudioService audioService;

  const PlaybackControlButton({required this.audioService, super.key});

  @override
  Widget build(BuildContext context) {
    return StreamBuilder<PlayerState>(
      stream: audioService.player.playerStateStream,
      builder: (context, snapshot) {
        final processingState = snapshot.data?.processingState;
        final playing = snapshot.data?.playing;
        return Row(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            IconButton(
              icon: const Icon(Icons.skip_previous),
              iconSize: 48,
              onPressed: audioService.playPrevious,
            ),
            if (processingState == ProcessingState.loading ||
                processingState == ProcessingState.buffering)
              Container(
                margin: const EdgeInsets.all(8.0),
                width: 64,
                height: 64,
                child: const CircularProgressIndicator(),
              )
            else if (playing != true)
              IconButton(
                icon: const Icon(Icons.play_arrow),
                iconSize: 64,
                onPressed: audioService.player.play,
              )
            else if (processingState != ProcessingState.completed)
              IconButton(
                icon: const Icon(Icons.pause),
                iconSize: 64,
                onPressed: audioService.player.pause,
              )
            else
              IconButton(
                icon: const Icon(Icons.replay),
                iconSize: 64,
                onPressed: () => audioService.player.seek(Duration.zero),
              ),
            IconButton(
              icon: const Icon(Icons.skip_next),
              iconSize: 48,
              onPressed: audioService.playNext,
            ),
          ],
        );
      },
    );
  }
}
