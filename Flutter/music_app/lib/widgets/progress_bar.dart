import 'package:flutter/material.dart';
import 'package:audio_video_progress_bar/audio_video_progress_bar.dart';
import '../services/audio_service.dart';

class ProgressBarWidget extends StatelessWidget {
  final AudioService audioService;

  const ProgressBarWidget({required this.audioService, super.key});

  @override
  Widget build(BuildContext context) {
    return StreamBuilder<Duration?>(
      stream: audioService.player.positionStream,
      builder: (context, snapshot) {
        return ProgressBar(
          progress: snapshot.data ?? Duration.zero,
          buffered: audioService.player.bufferedPosition,
          total: audioService.player.duration ?? Duration.zero,
          onSeek: (duration) {
            audioService.player.seek(duration);
          },
        );
      },
    );
  }
}
