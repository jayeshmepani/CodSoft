// widgets/recipe_card.dart
import 'package:flutter/material.dart';
import '../models/recipe.dart';

class RecipeCard extends StatelessWidget {
  final Recipe recipe;
  final Function() onPressed;

  RecipeCard({required this.recipe, required this.onPressed});

  @override
  Widget build(BuildContext context) {
    return Card(
      child: ListTile(
        title: Text(recipe.title),
        subtitle: Text(recipe.isDrink ? 'Drink' : 'Eat'), // Display if it's a drink or eat
        trailing: recipe.isVegetarian ? Image.asset('assets/img/veg.png') : Image.asset('assets/img/non-veg.png'),
        onTap: onPressed,
      ),
    );
  }
}
