// models/recipe.dart
class Recipe {
  final String title;
  final List<String> ingredients;
  final List<String> instructions;
  final bool isVegetarian;
  final bool isDrink; // New property

  Recipe({
    required this.title,
    required this.ingredients,
    required this.instructions,
    required this.isVegetarian,
    required this.isDrink, // Initialize in the constructor
  });
}
