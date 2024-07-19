// lib/screens/recipe_list.dart
import 'package:flutter/material.dart';
import '../models/recipe.dart';
import '../widgets/recipe_card.dart';
import 'recipe_details.dart';

class RecipeListScreen extends StatefulWidget {
  final List<Recipe> recipes;

  RecipeListScreen({required this.recipes});

  @override
  _RecipeListScreenState createState() => _RecipeListScreenState();
}

class _RecipeListScreenState extends State<RecipeListScreen> {
  String _filterType = 'All';
  String _sortType = 'None';
  TextEditingController _searchController = TextEditingController();

  @override
  void initState() {
    super.initState();
    _searchController.addListener(_filterRecipes);
  }

  @override
  void dispose() {
    _searchController.removeListener(_filterRecipes);
    _searchController.dispose();
    super.dispose();
  }

  void _filterRecipes() {
    setState(() {});
  }

  List<Recipe> get _filteredRecipes {
    List<Recipe> filteredRecipes = widget.recipes;

    if (_searchController.text.isNotEmpty) {
      filteredRecipes = filteredRecipes
          .where((recipe) => recipe.title.toLowerCase().contains(_searchController.text.toLowerCase()))
          .toList();
    }

    if (_filterType == 'Vegetarian') {
      filteredRecipes = filteredRecipes.where((recipe) => recipe.isVegetarian).toList();
    } else if (_filterType == 'Non-Vegetarian') {
      filteredRecipes = filteredRecipes.where((recipe) => !recipe.isVegetarian).toList();
    } else if (_filterType == 'Eat') {
      filteredRecipes = filteredRecipes.where((recipe) => !recipe.isDrink).toList();
    } else if (_filterType == 'Drink') {
      filteredRecipes = filteredRecipes.where((recipe) => recipe.isDrink).toList();
    }

    return filteredRecipes;
  }

  List<Recipe> get _sortedRecipes {
    List<Recipe> sortedRecipes = _filteredRecipes;
    if (_sortType == 'Name Ascending') {
      sortedRecipes.sort((a, b) => a.title.compareTo(b.title));
    } else if (_sortType == 'Name Descending') {
      sortedRecipes.sort((a, b) => b.title.compareTo(a.title));
    }
    return sortedRecipes;
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Recipes'),
        actions: [
          DropdownButton<String>(
            value: _filterType,
            onChanged: (String? newValue) {
              setState(() {
                _filterType = newValue!;
              });
            },
            items: <String>['All', 'Vegetarian', 'Non-Vegetarian', 'Eat', 'Drink']
                .map<DropdownMenuItem<String>>((String value) {
              return DropdownMenuItem<String>(
                value: value,
                child: Text(value),
              );
            }).toList(),
          ),
          DropdownButton<String>(
            value: _sortType,
            onChanged: (String? newValue) {
              setState(() {
                _sortType = newValue!;
              });
            },
            items: <String>['None', 'Name Ascending', 'Name Descending']
                .map<DropdownMenuItem<String>>((String value) {
              return DropdownMenuItem<String>(
                value: value,
                child: Text(value),
              );
            }).toList(),
          ),
        ],
      ),
      body: Column(
        children: [
          Padding(
            padding: const EdgeInsets.all(8.0),
            child: TextField(
              controller: _searchController,
              decoration: InputDecoration(
                labelText: 'Search Recipes',
                border: OutlineInputBorder(),
                suffixIcon: Icon(Icons.search),
              ),
            ),
          ),
          Expanded(
            child: ListView.builder(
              itemCount: _sortedRecipes.length,
              itemBuilder: (context, index) {
                return RecipeCard(
                  recipe: _sortedRecipes[index],
                  onPressed: () {
                    Navigator.push(
                      context,
                      MaterialPageRoute(
                        builder: (context) => RecipeDetailsScreen(recipe: _sortedRecipes[index]),
                      ),
                    );
                  },
                );
              },
            ),
          ),
        ],
      ),
    );
  }
}
