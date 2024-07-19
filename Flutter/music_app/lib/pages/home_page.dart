import 'package:flutter/material.dart';
import '../services/audio_service.dart';
import '../widgets/source_select.dart';
import '../widgets/progress_bar.dart';
import '../widgets/control_buttons.dart';
import '../widgets/playback_control_button.dart';

class HomePage extends StatefulWidget {
  const HomePage({super.key});

  @override
  State<HomePage> createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
  final AudioService _audioService = AudioService();

  @override
  void initState() {
    super.initState();
    WidgetsFlutterBinding.ensureInitialized();
    _audioService.setupAudioPlayer();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text("Music Player"),
      ),
      body: SafeArea(
        child: Padding(
          padding: const EdgeInsets.symmetric(horizontal: 16.0),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.center,
            mainAxisAlignment: MainAxisAlignment.start,
            children: [
              const SizedBox(height: 20),
              SourceSelect(audioService: _audioService),
              const SizedBox(height: 20),
              ProgressBarWidget(audioService: _audioService),
              const SizedBox(height: 20),
              ControlButtons(audioService: _audioService),
              const SizedBox(height: 20),
              PlaybackControlButton(audioService: _audioService),
            ],
          ),
        ),
      ),
    );
  }
}
