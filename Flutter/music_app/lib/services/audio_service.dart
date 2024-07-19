import 'package:just_audio/just_audio.dart';
import '../models/track.dart';

enum AudioSourceOption { Network, Asset }

class AudioService {
  final AudioPlayer _player = AudioPlayer();
  final List<Track> _playlist = [
    Track(
      title: "Aaj Mare Orde Re",
      artist: "Unknown",
      assetPath: "assets/audio/Aaj Mare Orde Re Full_(Peaceful).mp3",
    ),
    Track(
      title: "Pee Loon Ishq Sufiyana",
      artist: "Neha Kakkar, Sreerama",
      assetPath: "assets/audio/Pee_Loon_Ishq_Sufiyana_T-Series_Mixtape_Neha_Kakkar_Sreerama_Bhushan_Kumar_Ahmed_K_Abhijit_V.mp3",
    ),
    Track(
      title: "Gazab Ka Hai Din Bawara Mann",
      artist: "Shaan, Sukriti K",
      assetPath: "assets/audio/T-Series_Mixtape_-Gazab_Ka_Hai_Din_Bawara_Mann_Song_Shaan_Sukriti_K_Bhushan_Kumar_Ahmed_Abhijit.mp3",
    ),
  ];

  AudioPlayer get player => _player;
  List<Track> get playlist => _playlist;

  Future<void> setupAudioPlayer() async {
    _player.playbackEventStream.listen((event) {}, onError: (Object e, StackTrace stacktrace) {
      print("A stream error occurred: $e");
    });
    await loadAudioSource(_playlist[0]);
  }

  Future<void> loadAudioSource(Track track) async {
    try {
      await _player.setAudioSource(AudioSource.asset(track.assetPath));
    } catch (e) {
      print("Error loading audio source: $e");
    }
  }

  Future<void> playNext() async {
    var currentIndex = _playlist.indexOf(_playlist.firstWhere((track) => track.assetPath == _player.audioSource?.sequence[0].tag.toString()));
    if (currentIndex != -1 && currentIndex < _playlist.length - 1) {
      await loadAudioSource(_playlist[currentIndex + 1]);
      _player.play();
    }
  }

  Future<void> playPrevious() async {
    var currentIndex = _playlist.indexOf(_playlist.firstWhere((track) => track.assetPath == _player.audioSource?.sequence[0].tag.toString()));
    if (currentIndex > 0) {
      await loadAudioSource(_playlist[currentIndex - 1]);
      _player.play();
    }
  }
}
