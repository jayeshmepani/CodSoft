import 'package:flutter/material.dart';
import 'models/recipe.dart';
import 'screens/recipe_list.dart';

void main() {
  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Recipe App',
      theme: ThemeData(
        primarySwatch: Colors.blue,
        visualDensity: VisualDensity.adaptivePlatformDensity,
      ),
      home: RecipeListScreen(
        recipes: [
          Recipe(
            title: 'Spaghetti Carbonara',
            ingredients: [
              '200g spaghetti',
              '100g pancetta',
              '2 eggs',
              '50g grated parmesan cheese',
              'Salt and pepper to taste',
            ],
            instructions: [
              'Cook spaghetti according to package instructions.',
              'Fry pancetta until crispy.',
              'In a bowl, whisk eggs and grated parmesan cheese.',
              'Combine cooked spaghetti, crispy pancetta, and egg mixture.',
              'Season with salt and pepper.',
              'Serve hot.',
            ],
            isVegetarian: false,
            isDrink: false,
          ),
          Recipe(
            title: 'Chicken Alfredo',
            ingredients: [
              '250g fettuccine pasta',
              '2 chicken breasts, diced',
              '1 cup heavy cream',
              '1/2 cup grated parmesan cheese',
              '2 cloves garlic, minced',
              'Salt and pepper to taste',
            ],
            instructions: [
              'Cook fettuccine pasta according to package instructions.',
              'In a pan, cook diced chicken until golden brown.',
              'Add minced garlic and sauté until fragrant.',
              'Pour in heavy cream and simmer for a few minutes.',
              'Stir in grated parmesan cheese until melted and sauce thickens.',
              'Season with salt and pepper.',
              'Combine cooked pasta with the sauce.',
              'Serve hot.',
            ],
            isVegetarian: false,
            isDrink: false,
          ),
          Recipe(
            title: 'Vegetable Stir-Fry',
            ingredients: [
              '1 cup broccoli florets',
              '1 cup sliced bell peppers',
              '1 cup sliced carrots',
              '1 cup snap peas',
              '1/2 cup soy sauce',
              '2 tablespoons olive oil',
              '2 cloves garlic, minced',
              '1 teaspoon ginger, grated',
              'Salt and pepper to taste',
            ],
            instructions: [
              'Heat olive oil in a pan or wok over medium-high heat.',
              'Add minced garlic and grated ginger, sauté until fragrant.',
              'Add broccoli, bell peppers, carrots, and snap peas to the pan.',
              'Stir-fry vegetables until tender yet crisp.',
              'Pour in soy sauce and stir well to combine.',
              'Season with salt and pepper.',
              'Serve hot over cooked rice or noodles.',
            ],
            isVegetarian: true,
            isDrink: false,
          ),
          Recipe(
            title: 'Veg Biryani',
            ingredients: [
              '1 cup basmati rice',
              '1 onion, sliced',
              '1 tomato, chopped',
              '1/2 cup mixed vegetables (peas, carrots, beans)',
              '1/4 cup yogurt',
              '1/4 cup fried onions',
              '1 teaspoon ginger-garlic paste',
              '1/2 teaspoon biryani masala',
              '1/2 teaspoon turmeric powder',
              '1/2 teaspoon red chili powder',
              '1/4 teaspoon garam masala',
              'Few strands of saffron soaked in milk',
              'Salt to taste',
            ],
            instructions: [
              'Soak basmati rice in water for 30 minutes, then drain.',
              'In a pan, heat oil and sauté onions until golden brown.',
              'Add ginger-garlic paste and chopped tomatoes, cook until tomatoes soften.',
              'Add mixed vegetables and cook until partially tender.',
              'Add biryani masala, turmeric powder, red chili powder, and salt.',
              'Add yogurt and mix well.',
              'Layer soaked rice over the vegetable mixture.',
              'Sprinkle garam masala and fried onions on top.',
              'Pour saffron-infused milk over the rice.',
              'Cover and cook on low heat until rice is fully cooked and aromatic.',
              'Serve hot with raita.',
            ],
            isVegetarian: true,
            isDrink: false,
          ),
          Recipe(
            title: 'Pulao',
            ingredients: [
              '1 cup basmati rice',
              '1 onion, sliced',
              '1 carrot, diced',
              '1 potato, diced',
              '1/2 cup peas',
              '1/4 cup cashews',
              '2 tablespoons ghee',
              '1 teaspoon cumin seeds',
              '1 bay leaf',
              '1 cinnamon stick',
              '2 cloves',
              '2 cups water',
              'Salt to taste',
            ],
            instructions: [
              'Wash and soak basmati rice in water for 30 minutes.',
              'In a pressure cooker or pan, heat ghee and sauté cashews until golden brown.',
              'Add cumin seeds, bay leaf, cinnamon stick, and cloves, sauté for a minute.',
              'Add sliced onions and sauté until onions turn translucent.',
              'Add diced carrots, potatoes, and peas, sauté for a few minutes.',
              'Drain soaked rice and add it to the cooker.',
              'Add water and salt, stir well.',
              'Pressure cook for 2 whistles or cook covered until rice is done.',
              'Fluff the rice with a fork and serve hot.',
            ],
            isVegetarian: true,
            isDrink: false,
          ),
          Recipe(
            title: 'Pizza',
            ingredients: [
              '1 pre-made pizza crust',
              '1/2 cup pizza sauce',
              '1 cup shredded mozzarella cheese',
              '1/2 cup sliced bell peppers',
              '1/2 cup sliced mushrooms',
              '1/4 cup sliced black olives',
              '1/4 cup chopped onions',
              '1/4 cup chopped tomatoes',
              '1/2 teaspoon dried oregano',
              '1/2 teaspoon dried basil',
              'Salt and pepper to taste',
            ],
            instructions: [
              'Preheat oven according to pizza crust instructions.',
              'Spread pizza sauce evenly over the crust.',
              'Sprinkle shredded mozzarella cheese on top of the sauce.',
              'Arrange sliced vegetables (bell peppers, mushrooms, olives, onions, tomatoes) over the cheese.',
              'Sprinkle dried oregano, dried basil, salt, and pepper.',
              'Bake in the preheated oven until the cheese melts and crust turns golden brown.',
              'Slice and serve hot.',
            ],
            isVegetarian: true,
            isDrink: false,
          ),
          Recipe(
            title: 'Creamy Mushroom Pasta',
            ingredients: [
              '200g pasta of your choice',
              '1 cup sliced mushrooms',
              '1/2 cup heavy cream',
              '1/4 cup grated parmesan cheese',
              '2 cloves garlic, minced',
              '2 tablespoons butter',
              '1 tablespoon olive oil',
              'Salt and pepper to taste',
              'Fresh parsley for garnish',
            ],
            instructions: [
              'Cook pasta according to package instructions, drain and set aside.',
              'In a pan, heat butter and olive oil.',
              'Add minced garlic and sauté until fragrant.',
              'Add sliced mushrooms and cook until they release their moisture.',
              'Pour in heavy cream and bring to a simmer.',
              'Add grated parmesan cheese, salt, and pepper, stir until cheese melts and sauce thickens.',
              'Add cooked pasta to the sauce, toss well to coat.',
              'Garnish with fresh parsley and serve hot.',
            ],
            isVegetarian: true,
            isDrink: false,
          ),
          Recipe(
            title: 'Spinach and Cheese Stuffed Mushrooms',
            ingredients: [
              '8 large mushrooms, cleaned and stems removed',
              '1 cup chopped spinach',
              '1/2 cup ricotta cheese',
              '1/4 cup grated mozzarella cheese',
              '1/4 cup grated parmesan cheese',
              '2 cloves garlic, minced',
              '2 tablespoons olive oil',
              'Salt and pepper to taste',
            ],
            instructions: [
              'Preheat oven to 375°F (190°C).',
              'In a pan, heat olive oil and sauté minced garlic until fragrant.',
              'Add chopped spinach and cook until wilted.',
              'In a bowl, mix together ricotta cheese, mozzarella cheese, parmesan cheese, sautéed spinach, salt, and pepper.',
              'Stuff each mushroom cap with the cheese and spinach mixture.',
              'Place stuffed mushrooms on a baking sheet.',
              'Bake in the preheated oven for 15-20 minutes or until mushrooms are tender and cheese is melted.',
              'Serve hot as an appetizer or side dish.',
            ],
            isVegetarian: true,
            isDrink: false,
          ),
          Recipe(
            title: 'Mango Lassi',
            ingredients: [
              '1 cup ripe mango chunks',
              '1 cup plain yogurt',
              '1/2 cup milk',
              '2 tablespoons sugar (or to taste)',
              'Pinch of cardamom powder (optional)',
              'Ice cubes',
            ],
            instructions: [
              'Combine mango, yogurt, milk, sugar, and cardamom (if using) in a blender.',
              'Blend until smooth and creamy.',
              'Add ice cubes and blend again until frothy.',
              'Serve chilled.'
            ],
            isVegetarian: true,
            isDrink: true,
          ),
          Recipe(
            title: 'Vegetarian Lasagna',
            ingredients: [
              '9 lasagna noodles',
              '2 tablespoons olive oil',
              '1 onion, chopped',
              '2 cloves garlic, minced',
              '1 red bell pepper, chopped',
              '1 zucchini, chopped',
              '1 (28-ounce) can crushed tomatoes',
              '1 (6-ounce) can tomato paste',
              '1 teaspoon dried oregano',
              '1/2 teaspoon dried basil',
              'Salt and pepper to taste',
              '15 ounces ricotta cheese',
              '1/2 cup grated Parmesan cheese',
              '1 egg',
              '10 ounces fresh spinach, chopped',
              '8 ounces mozzarella cheese, shredded',
            ],
            instructions: [
              'Preheat oven to 375°F (190°C).',
              'Cook lasagna noodles according to package directions. Drain and rinse with cold water.',
              'Heat olive oil in a large skillet over medium heat. Add onion, garlic, bell pepper, and zucchini. Cook until softened.',
              'Stir in crushed tomatoes, tomato paste, oregano, basil, salt, and pepper. Bring to a simmer and cook for 15 minutes.',
              'In a bowl, combine ricotta cheese, Parmesan cheese, egg, and spinach.',
              'Spread a thin layer of tomato sauce in the bottom of a 9x13 inch baking dish.',
              'Top with a layer of lasagna noodles, ricotta mixture, and tomato sauce. Repeat layers, ending with tomato sauce and mozzarella cheese.',
              'Bake for 25-30 minutes, or until golden brown and bubbly.',
              'Let rest for 10 minutes before serving.'
            ],
            isVegetarian: true,
            isDrink: false,
          ),
          Recipe(
            title: 'Masala Chai',
            ingredients: [
              '2 cups water',
              '2 teaspoons loose black tea leaves (or 2 tea bags)',
              '1/2 cup milk',
              '1/2 teaspoon ground ginger',
              '1/4 teaspoon ground cinnamon',
              '1/4 teaspoon ground cardamom',
              'Pinch of ground cloves',
              'Sugar to taste',
            ],
            instructions: [
              'Combine water, tea leaves (or tea bags), and spices in a saucepan.',
              'Bring to a boil, then reduce heat and simmer for 5 minutes.',
              'Add milk and sugar to taste. Bring to a gentle boil.',
              'Strain into cups and serve hot.'
            ],
            isVegetarian: true,
            isDrink: true,
          ),
          Recipe(
            title: 'Aperol Spritz',
            ingredients: [
              '3 parts Prosecco',
              '2 parts Aperol',
              '1 part soda water',
              'Ice cubes',
              'Orange slice for garnish',
            ],
            instructions: [
              'Fill a wine glass with ice.',
              'Pour in Prosecco, Aperol, and soda water.',
              'Garnish with an orange slice and enjoy.'
            ],
            isVegetarian: true,
            isDrink: true,
          ),
          Recipe(
            title: 'French Press Coffee',
            ingredients: [
              'Coarsely ground coffee (3 tablespoons per 1 cup of water)',
              'Hot water (just off the boil)',
            ],
            instructions: [
              'Add coffee grounds to the French press.',
              'Pour hot water over the grounds, filling the press about 3/4 full.',
              'Stir gently to ensure all grounds are saturated.',
              'Place the lid on the press (without plunging yet) and let steep for 4 minutes.',
              'Slowly press down the plunger.',
              'Pour and enjoy immediately.'
            ],
            isVegetarian: true,
            isDrink: true,
          ),
          Recipe(
            title: 'Espresso',
            ingredients: [
              '7-8 grams finely ground coffee beans',
              '1-2 ounces filtered water (heated to 195-205°F)'
            ],
            instructions: [
              'Grind the coffee beans to a fine consistency.',
              'Place the ground coffee in a portafilter and tamp firmly.',
              'Insert the portafilter into an espresso machine.',
              'Brew for 25-30 seconds, until you have 1-2 ounces of espresso.',
              'Serve immediately in a small cup.'
            ],
            isVegetarian: true,
            isDrink: true,
          ),
          Recipe(
            title: 'Cappuccino',
            ingredients: [
              '1/3 cup freshly brewed espresso',
              '1/3 cup steamed milk',
              '1/3 cup milk foam'
            ],
            instructions: [
              'Prepare a single or double shot of espresso.',
              'Steam milk until it is hot and foamy.',
              'Pour the espresso into a cup.',
              'Slowly pour the steamed milk into the cup, holding back the foam with a spoon.',
              'Top with the remaining foam.'
            ],
            isVegetarian: true,
            isDrink: true,
          ),
          Recipe(
            title: 'Latte',
            ingredients: [
              '1/3 cup freshly brewed espresso',
              '2/3 cup steamed milk',
              'A small amount of milk foam (optional)'
            ],
            instructions: [
              'Prepare a single or double shot of espresso.',
              'Steam milk until it is hot and slightly frothy.',
              'Pour the espresso into a cup.',
              'Slowly pour the steamed milk into the cup.',
              'If desired, top with a small amount of foam.'
            ],
            isVegetarian: true,
            isDrink: true,
          ),
          Recipe(
            title: 'Americano',
            ingredients: ['1 shot of espresso', '6-8 ounces hot water'],
            instructions: [
              'Prepare a single or double shot of espresso.',
              'Pour the espresso into a cup.',
              'Add hot water to desired strength.'
            ],
            isVegetarian: true,
            isDrink: true,
          ),
          Recipe(
            title: 'Mocha',
            ingredients: [
              '1/4 cup chocolate syrup or powder',
              '1/3 cup freshly brewed espresso',
              '2/3 cup steamed milk',
              'Whipped cream (optional)'
            ],
            instructions: [
              'Add chocolate syrup or powder to a cup.',
              'Prepare a single or double shot of espresso and pour into the cup.',
              'Steam milk until it is hot and frothy.',
              'Pour the steamed milk into the cup.',
              'If desired, top with whipped cream.'
            ],
            isVegetarian: true,
            isDrink: true,
          ),
          Recipe(
            title: 'Dosa',
            ingredients: [
              '1 cup parboiled rice',
              '1/2 cup urad dal (split black lentils)',
              '1/4 teaspoon fenugreek seeds',
              'Salt to taste',
              'Oil for cooking',
            ],
            instructions: [
              'Soak rice, urad dal, and fenugreek seeds separately for 4-6 hours.',
              'Grind each ingredient separately into asmooth batter.',
              'Mix the batters together, add salt, and let ferment for 8-10 hours.',
              'Heat a non-stick pan, spread a ladleful of batter in a circular motion.',
              'Drizzle oil and cook until golden brown on both sides.',
              'Serve hot with chutney or sambar.'
            ],
            isVegetarian: true,
            isDrink: false,
          ),
          Recipe(
            title: 'Idli Sambar',
            ingredients: [
              'For Idli:',
              '1 cup parboiled rice',
              '1/2 cup urad dal (split black lentils)',
              'Salt to taste',
              'For Sambar:',
              '1 cup toor dal (pigeon peas)',
              'Mixed vegetables (drumsticks, carrots, pumpkin, etc.)',
              '1 tablespoon sambar powder',
              '1/2 teaspoon turmeric powder',
              '1 teaspoon mustard seeds',
              '1 teaspoon cumin seeds',
              'A few curry leaves',
              '2 dried red chilies',
              '1 tablespoon oil',
              'Salt to taste',
              'Tamarind pulp',
            ],
            instructions: [
              ' smooth batter.',
              'Mix the batters together, add salt, and let ferment for 8-10 hours.',
              'Steam the batter in idli molds until cooked.',
              'For Sambar:',
              'Pressure cook toor dal with turmeric powder until soft.',
              'In a pot, temper mustard seeds, cumin seeds, curry leaves, and red chilies in oil.',
              'Add chopped vegetables and sambar powder. Cook until vegetables are tender.',
              'Add cooked dal, tamarind pulp, and salt. Simmer for 10-15 minutes.',
              'Serve hot idli with sambar.'
            ],
            isVegetarian: true,
            isDrink: false,
          ),
          Recipe(
            title: 'Upma',
            ingredients: [
              '1 cup semolina (rava/sooji)',
              '2 tablespoons ghee or oil',
              '1 teaspoon mustard seeds',
              '1 teaspoon cumin seeds',
              'A few curry leaves',
              '1 green chili, chopped',
              '1 onion, chopped',
              '1/2 cup mixed vegetables (peas, carrots, beans), chopped',
              '2 cups water',
              'Salt to taste',
              'Chopped cilantro for garnish',
            ],
            instructions: [
              'Dry roast semolina until fragrant.',
              'In a pot, heat ghee or oil. Add mustard seeds, cumin seeds, curry leaves, and green chili.',
              'Add onion and sauté until golden brown. Add vegetables and cook until tender.',
              'Add roasted semolina and water. Stir continuously to avoid lumps.',
              'Cook until upma thickens and is cooked through.',
              'Garnish with cilantro and serve hot.'
            ],
            isVegetarian: true,
            isDrink: false,
          ),
          Recipe(
            title: 'Uttapam',
            ingredients: [
              '1 cup dosa batter (fermented rice and lentil batter)',
              '1/4 cup chopped onion',
              '1/4 cup chopped tomato',
              '1 green chili, chopped',
              '1/4 cup chopped cilantro',
              'Oil for cooking',
              'Salt to taste',
            ],
            instructions: [
              'Heat a non-stick pan, spread a ladleful of dosa batter in a circular motion.',
              'Sprinkle chopped onion, tomato, green chili, and cilantro on top.',
              'Drizzle oil and cook until golden brown on the bottom.',
              'Flip and cook the other side until golden brown.',
              'Serve hot with chutney or sambar.'
            ],
            isVegetarian: true,
            isDrink: false,
          ),
          Recipe(
            title: 'Aloo Paratha',
            ingredients: [
              'For dough:',
              '2 cups whole wheat flour',
              'Water for kneading',
              'Salt to taste',
              'For filling:',
              '2-3 boiled potatoes, mashed',
              '1 onion, finely chopped',
              '1 green chili, chopped',
              '1/2 teaspoon cumin seeds',
              '1/2 teaspoon coriander powder',
              '1/4 teaspoon red chili powder',
              '1/4 teaspoon amchur (dried mango powder)',
              'Chopped cilantro',
              'Salt to taste',
              'Ghee or oil for cooking',
            ],
            instructions: [
              'Prepare dough by kneading flour with water and salt.',
              'Mix all filling ingredients together.',
              'Divide dough into small balls. Roll each ball into a circle.',
              'Place a portion of filling in the center, bring the edges together, and seal.',
              'Roll out the stuffed ball into a paratha.',
              'Cook on a hot griddle with ghee or oil until golden brown on both sides.',
              'Serve hot with yogurt or pickle.'
            ],
            isVegetarian: true,
            isDrink: false,
          ),
          Recipe(
            title: 'Chilla (Pudla)',
            ingredients: [
              '1 cup besan (gram flour)',
              '1/4 cup finely chopped onion',
              '1/4 cup finely chopped tomatoes',
              '1 green chili, finely chopped',
              '1/2 teaspoon grated ginger',
              'A pinch of turmeric powder',
              'A pinch of red chili powder',
              '1/4 teaspoon ajwain (carom seeds)',
              'Chopped cilantro',
              'Salt to taste',
              'Water (as needed)',
              'Oil for cooking'
            ],
            instructions: [
              'In a bowl, mix besan, chopped vegetables, spices, and salt.',
              'Add water gradually to make a smooth batter, not too thick or thin.',
              'Heat a non-stick pan. Pour a ladleful of batter and spread it into a thin circle.',
              'Drizzle oil and cook until golden brown. Flip and cook the other side.',
              'Serve hot with chutney or yogurt.'
            ],
            isVegetarian: true,
            isDrink: false,
          ),
          Recipe(
            title: 'Paneer Tikka',
            ingredients: [
              '250g paneer (Indian cottage cheese), cubed',
              '1/2 cup thick yogurt',
              '1 tablespoon ginger-garlic paste',
              '1 tablespoon lemon juice',
              '1 teaspoon turmeric powder',
              '1 teaspoon red chili powder',
              '1 teaspoon garam masala',
              '1 teaspoon kasoori methi (dried fenugreek leaves)',
              '1/2 teaspoon chaat masala',
              'Salt to taste',
              '2 tablespoons oil',
              '1 bell pepper, cubed',
              '1 onion, cubed'
            ],
            instructions: [
              'Marinate paneer, bell pepper, and onion in yogurt and spices for at least 30 minutes.',
              'Thread marinated ingredients onto skewers.',
              'Heat oil in a pan or grill. Cook skewers until paneer is golden brown and vegetables are tender.',
              'Serve hot with mint chutney and onion rings.'
            ],
            isVegetarian: true,
            isDrink: false,
          ),
          Recipe(
            title: 'Khaman Dhokla',
            ingredients: [
              '1 cup besan (gram flour)',
              '1/4 cup semolina (sooji)',
              '1 teaspoon turmeric powder',
              '1 teaspoon chili powder',
              '1 teaspoon ginger-green chili paste',
              '1/2 teaspoon citric acid',
              '1/4 teaspoon baking soda',
              'Salt to taste',
              '1 tablespoon sugar',
              '1/4 cup yogurt',
              'Water as needed',
              'For Tempering:',
              '2 tablespoons oil',
              '1 teaspoon mustard seeds',
              '1 teaspoon sesame seeds',
              '2-3 green chilies, slit',
              'A few curry leaves',
              '1 tablespoon grated coconut',
              'Chopped cilantro'
            ],
            instructions: [
              'Mix besan, semolina, turmeric, chili powder, ginger-green chili paste, citric acid, salt, and sugar.',
              'Add yogurt and water to make a smooth batter. Let it rest for 15 minutes.',
              'Add baking soda and mix well. Immediately pour the batter into a greased pan.',
              'Steam for 15-20 minutes until cooked.',
              'For tempering, heat oil, add mustard seeds, sesame seeds, green chilies, and curry leaves.',
              'Pour tempering over dhokla. Garnish with coconut and cilantro.'
            ],
            isVegetarian: true,
            isDrink: false,
          ),
          Recipe(
            title: 'Tacos al Pastor',
            ingredients: [
              '1 pound boneless pork shoulder, thinly sliced',
              '1/2 cup pineapple juice',
              '1/4 cup adobo sauce',
              '2 tablespoons chili powder',
              '1 tablespoon cumin',
              '1 teaspoon oregano',
              '1/2 teaspoon garlic powder',
              '1/4 teaspoon salt',
              '1/4 teaspoon black pepper',
              'Corn tortillas',
              'Chopped onions',
              'Chopped cilantro',
              'Pineapple chunks',
              'Salsa'
            ],
            instructions: [
              'Marinate pork in a mixture of pineapple juice, adobo sauce, chili powder, cumin, oregano, garlic powder, salt, and pepper.',
              'Cook marinated pork on a grill or skillet until browned and cooked through.',
              'Warm corn tortillas.',
              'Fill tortillas with pork, onions, cilantro, pineapple, and salsa.'
            ],
            isVegetarian: false,
            isDrink: false,
          ),
          Recipe(
            title: 'Chow Mein',
            ingredients: [
              '8 ounces egg noodles (or your preferred type)',
              '2 tablespoons vegetable oil',
              '1/2 pound protein of choice (chicken, beef, shrimp, or tofu), sliced',
              '1/2 onion, sliced',
              '1 carrot, julienned',
              '1/2 cup sliced cabbage',
              '1/4 cup sliced mushrooms',
              'For the sauce:',
              '2 tablespoons soy sauce',
              '1 tablespoon oyster sauce',
              '1 teaspoon sesame oil',
              '1/2 teaspoon sugar',
              '1/4 teaspoon white pepper',
            ],
            instructions: [
              'Cook noodles according to package directions. Drain and toss with a little oil to prevent sticking.',
              'Heat oil in a wok or skillet over high heat. Stir-fry protein until cooked through.',
              'Add vegetables and stir-fry until crisp-tender.',
              'Push ingredients to the side and pour in the sauce. Bring to a simmer.',
              'Add noodles and toss to coat in the sauce.',
              'Serve immediately.'
            ],
            isVegetarian: false,
            isDrink: false,
          ),
          Recipe(
            title: 'Eggnog',
            ingredients: [
              '6 large eggs, separated',
              '1/2 cup sugar',
              '1/4 teaspoon salt',
              '4 cups milk',
              '1 cup heavy cream',
              '1/4 cup bourbon, rum, or brandy (optional)',
              'Ground nutmeg for garnish'
            ],
            instructions: [
              'Beat egg yolks with sugar and salt until light and fluffy.',
              'Slowly whisk in milk and cream.',
              'In a separate bowl, beat egg whites until stiff peaks form.',
              'Gently fold egg whites into the yolk mixture.',
              'Stir in alcohol, if using.',
              'Chill for at least 1 hour.',
              'Serve in glasses, garnished with nutmeg.'
            ],
            isVegetarian: false,
            isDrink: true,
          ),
          Recipe(
            title: 'Bone Broth',
            ingredients: [
              '2 pounds beef or chicken bones',
              '1 tablespoon apple cider vinegar',
              '1 onion, quartered',
              '2 carrots, chopped',
              '2 celery stalks, chopped',
              'A few sprigs of fresh herbs (e.g., thyme, parsley)',
              '10 cups water',
              'Salt and pepper to taste'
            ],
            instructions: [
              'Place bones, vinegar, vegetables, and herbs in a large pot or slow cooker.',
              'Add water to cover the ingredients.',
              'Bring to a boil, then reduce heat to a simmer.',
              'Simmer for at least 12 hours (up to 24 hours for beef bones).',
              'Strain the broth and discard the solids.',
              'Season with salt and pepper to taste.',
              'Serve hot or use as a base for soups and stews.'
            ],
            isVegetarian: false,
            isDrink: true,
          ),
          Recipe(
            title: 'Vietnamese Egg Coffee (Cà Phê Trứng)',
            ingredients: [
              '1-2 tablespoons sweetened condensed milk',
              '1 egg yolk',
              '1/2 cup hot Vietnamese coffee (or strong brewed coffee)'
            ],
            instructions: [
              'In a small bowl, whisk together sweetened condensed milk and egg yolk until light and fluffy.',
              'Slowly pour hot coffee into the egg yolk mixture, whisking constantly to prevent curdling.',
              'Pour the mixture into a glass or cup.',
              'Enjoy warm!'
            ],
            isVegetarian: false,
            isDrink: true,
          ),
          Recipe(
            title: 'Mongolian Salted Milk Tea (Süütei Tsai)',
            ingredients: [
              '2 cups water',
              '2 teaspoons loose leaf green tea (preferably brick tea)',
              '1/4 teaspoon salt',
              '1/4 cup milk',
              '1 tablespoon butter (optional)'
            ],
            instructions: [
              'Bring water to a boil in a saucepan.',
              'Add green tea and salt. Reduce heat and simmer for 5 minutes.',
              'Strain the tea into a cup.',
              'Add milk and butter (if using).',
              'Stir and enjoy warm.'
            ],
            isVegetarian: false,
            isDrink: true,
          ),
          Recipe(
            title: 'Bulletproof Coffee',
            ingredients: [
              '1 cup freshly brewed coffee',
              '1 tablespoon unsalted butter or ghee',
              '1 tablespoon MCT oil (or coconut oil)'
            ],
            instructions: [
              'Brew your favorite coffee.',
              'Add butter and MCT oil to the coffee.',
              'Blend with a hand blender or in a blender until frothy and creamy.',
              'Enjoy your Bulletproof Coffee for a boost of energy and focus!'
            ],
            isVegetarian: false,
            isDrink: true,
          ),
          Recipe(
            title: 'Aam Panna (Raw Mango Drink)',
            ingredients: [
              '1 large raw mango',
              '1 cup sugar (adjust to taste)',
              '1 teaspoon roasted cumin powder',
              '1/2 teaspoon black salt',
              'A pinch of salt',
              '4 cups water',
              'Fresh mint leaves for garnish'
            ],
            instructions: [
              'Boil the raw mango until soft. Peel and remove the pulp.',
              'Blend the pulp with sugar, cumin powder, black salt, and salt.',
              'Add water and blend again until smooth.',
              'Strain the mixture and chill.',
              'Serve chilled, garnished with mint leaves.'
            ],
            isVegetarian: true,
            isDrink: true,
          ),
          Recipe(
            title: 'Thandai (Spiced Milk Drink)',
            ingredients: [
              '1/4 cup almonds',
              '1/4 cup cashews',
              '1/4 cup pistachios',
              '2 tablespoons poppy seeds',
              '1 tablespoon fennel seeds',
              '1 teaspoon watermelon seeds (optional)',
              '5-6 peppercorns',
              '4-5 green cardamom pods',
              '1/4 teaspoon saffron strands',
              '1/2 cup milk',
              '4 cups chilled milk',
              'Sugar to taste',
              'Rose water (optional)'
            ],
            instructions: [
              'Soak all the nuts, seeds, and spices in water for 2-3 hours.',
              'Drain and grind them into a smooth paste with 1/2 cup milk.',
              'Add this paste to the chilled milk and mix well.',
              'Add sugar to taste and a few drops of rose water (optional).',
              'Serve chilled, garnished with chopped nuts.'
            ],
            isVegetarian: true,
            isDrink: true,
          ),
          Recipe(
            title: 'Kir Royale',
            ingredients: [
              '1/2 ounce crème de cassis (blackcurrant liqueur)',
              'Chilled Champagne or dry sparkling wine',
              'Raspberry for garnish (optional)'
            ],
            instructions: [
              'Pour crème de cassis into a champagne flute.',
              'Top with chilled Champagne or sparkling wine.',
              'Garnish with a raspberry, if desired.'
            ],
            isVegetarian: true,
            isDrink: true,
          ),
          Recipe(
            title: 'Matcha Latte',
            ingredients: [
              '1 teaspoon matcha powder',
              '2 ounces hot water',
              '1 cup steamed milk (dairy or non-dairy)'
            ],
            instructions: [
              'Whisk matcha powder and hot water in a bowl until smooth and frothy.',
              'Steam milk until hot and frothy.',
              'Pour matcha mixture into a cup.',
              'Top with steamed milk.',
              'Enjoy the vibrant green latte!'
            ],
            isVegetarian: true,
            isDrink: true,
          ),
          Recipe(
            title: 'French 75',
            ingredients: [
              '1 1/2 ounces gin',
              '3/4 ounce lemon juice',
              '1/2 ounce simple syrup',
              'Chilled Champagne or dry sparkling wine',
              'Lemon twist for garnish'
            ],
            instructions: [
              'Combine gin, lemon juice, and simple syrup in a cocktail shaker with ice.',
              'Shake until well chilled.',
              'Strain into a champagne flute.',
              'Top with Champagne or sparkling wine.',
              'Garnish with a lemon twist.'
            ],
            isVegetarian: true,
            isDrink: true,
          ),
          Recipe(
            title: 'Pisco Sour',
            ingredients: [
              '2 ounces pisco (Peruvian brandy)',
              '1 ounce lime juice',
              '1 ounce simple syrup',
              '1 egg white',
              'Angostura bitters for garnish'
            ],
            instructions: [
              'Combine all ingredients except bitters in a cocktail shaker without ice (dry shake).',
              'Add ice and shake again until well-chilled.',
              'Strain into a chilled glass.',
              'Garnish with a few drops of Angostura bitters.'
            ],
            isVegetarian: false,
            isDrink: true,
          ),
          Recipe(
            title: 'Tres Leches Cake',
            ingredients: [
              '1 cup all-purpose flour',
              '1 1/2 teaspoons baking powder',
              '1/4 teaspoon salt',
              '1/2 cup unsalted butter, softened',
              '3/4 cup sugar',
              '5 eggs, separated',
              '1/2 cup milk',
              '1 teaspoon vanilla extract',
              'For the Tres Leches Mixture:',
              '1 can (14 ounces) sweetened condensed milk',
              '1 can (12 ounces) evaporated milk',
              '1 cup heavy cream',
              'Whipped cream and maraschino cherries for topping (optional)'
            ],
            instructions: [
              'Preheat oven to 350°F (175°C). Grease and flour a 9x13 inch baking pan.',
              'Whisk together flour, baking powder, and salt.',
              'Beat butter and sugar until light and fluffy.',
              'Beat in egg yolks one at a time, then stir in milk and vanilla.',
              'Gradually add flour mixture to wet ingredients, mixing until just combined.',
              'In a separate bowl, beat egg whites until stiff peaks form. Fold into batter.',
              'Pour batter into prepared pan and bake for 30-35 minutes, or until a toothpick inserted in the center comes out clean.',
              'While cake bakes, whisk together tres leches ingredients.',
              'Once cake is cool, poke holes all over with a fork.',
              'Pour tres leches mixture over cake and refrigerate for at least 4 hours or overnight.',
              'Top with whipped cream and cherries before serving, if desired.'
            ],
            isVegetarian: true,
            isDrink: false,
          ),
          Recipe(
            title: 'Churros',
            ingredients: [
              '1 cup water',
              '1/2 cup unsalted butter',
              '1/4 teaspoon salt',
              '1 cup all-purpose flour',
              '2 eggs',
              'Oil for frying',
              '1/2 cup sugar',
              '1 teaspoon ground cinnamon',
            ],
            instructions: [
              'In a saucepan, combine water, butter, and salt. Bring to a boil.',
              'Remove from heat and stir in flour until a dough forms.',
              'Beat in eggs one at a time until incorporated.',
              'Heat oil for frying.',
              'Pipe or spoon dough into hot oil and fry until golden brown.',
              'Drain on paper towels.',
              'Roll in cinnamon sugar while still warm.'
            ],
            isVegetarian: true,
            isDrink: false,
          ),
          Recipe(
            title: 'Rasmalai',
            ingredients: [
              'For Chenna (Cheese Balls):',
              '1 liter whole milk',
              '1/4 cup lemon juice or vinegar',
              'For Sugar Syrup:',
              '2 cups sugar',
              '4 cups water',
              'A few cardamom pods',
              'A few strands of saffron',
              'For Ras (Milk Sauce):',
              '1 liter milk',
              '1/2 cup sugar',
              'A few cardamom pods',
              'A few strands of saffron',
              'Chopped pistachios for garnish'
            ],
            instructions: [
              'To make chenna, boil milk, add lemon juice, and stir until milk curdles. Strain through a cheesecloth.',
              'Knead the chenna for a few minutes, then divide into small balls.',
              'Make sugar syrup by boiling sugar, water, cardamom, and saffron until it thickens slightly.',
              'Flatten the chenna balls and cook in the sugar syrup for 10-15 minutes until they puff up.',
              'Make ras by boiling milk, sugar, cardamom, and saffron until slightly reduced.',
              'Let the chenna balls cool slightly, then add them to the ras.',
              'Chill and serve garnished with chopped pistachios.'
            ],
            isVegetarian: true,
            isDrink: false,
          ),
          Recipe(
            title: 'Gulab Jamun',
            ingredients: [
              '1 cup milk powder (khoya)',
              '1/4 cup all-purpose flour (maida)',
              '1/4 teaspoon baking soda',
              '1 tablespoon ghee',
              'Milk (for kneading)',
              'Oil for deep frying',
              'For Sugar Syrup:',
              '2 cups sugar',
              '1 cup water',
              'A few cardamom pods',
              'A few strands of saffron'
            ],
            instructions: [
              'Combine milk powder, flour, baking soda, and ghee. Knead with milk to form a soft dough.',
              'Shape dough into small balls.',
              'Heat oil in a deep pan. Deep-fry the balls over low heat until golden brown.',
              'Prepare sugar syrup by boiling sugar, water, cardamom, and saffron until it thickens slightly.',
              'Add the fried gulab jamun to the sugar syrup and let them soak for at least 2 hours.',
              'Serve warm or at room temperature.'
            ],
            isVegetarian: true,
            isDrink: false,
          ),
          Recipe(
            title: 'Classic Vanilla Cupcakes',
            ingredients: [
              '1 1/2 cups all-purpose flour',
              '1 1/2 teaspoons baking powder',
              '1/2 teaspoon salt',
              '1 cup granulated sugar',
              '1/2 cup unsalted butter, softened',
              '2 large eggs',
              '2 teaspoons vanilla extract',
              '1/2 cup milk',
            ],
            instructions: [
              'Preheat oven to 350°F (175°C). Line a muffin tin with cupcake liners.',
              'In a bowl, whisk together flour, baking powder, and salt.',
              'In a separate bowl, cream together sugar and butter until light and fluffy.',
              'Beat in eggs one at a time, then stir in vanilla extract.',
              'Gradually add the dry ingredients to the wet ingredients, alternating with milk. Begin and end with dry ingredients.',
              'Divide batter among cupcake liners.',
              'Bake for 18-22 minutes, or until a toothpick inserted into the center comes out clean.',
              'Let cool completely before frosting.'
            ],
            isVegetarian: true,
            isDrink: false,
          ),
          Recipe(
            title: 'New York Cheesecake',
            ingredients: [
              'For the crust:',
              '1 1/2 cups graham cracker crumbs',
              '5 tablespoons unsalted butter, melted',
              '1/4 cup granulated sugar',
              'For the filling:',
              '32 ounces cream cheese, softened',
              '1 1/4 cups granulated sugar',
              '1/4 cup sour cream',
              '3 tablespoons all-purpose flour',
              '1 teaspoon vanilla extract',
              '1/2 teaspoon lemon zest',
              '4 large eggs',
            ],
            instructions: [
              'Preheat oven to 350°F (175°C).',
              'Combine graham cracker crumbs, melted butter, and sugar for the crust. Press into the bottom of a 9-inch springform pan.',
              'Bake for 10 minutes. Let cool completely.',
              'Beat cream cheese and sugar until smooth. Add sour cream, flour, vanilla, and lemon zest; beat until combined.',
              'Beat in eggs one at a time, just until incorporated.',
              'Pour filling over the crust.',
              'Bake for 1 hour and 15 minutes, or until the center is almost set. Turn off oven and let cheesecake sit in the oven for 1 hour.',
              'Remove from oven and cool completely. Refrigerate for at least 4 hours before serving.'
            ],
            isVegetarian: true,
            isDrink: false,
          ),
          Recipe(
            title: 'No-Churn Vanilla Ice Cream',
            ingredients: [
              '2 cups heavy cream',
              '1 can (14 ounces) sweetened condensed milk',
              '2 teaspoons vanilla extract',
            ],
            instructions: [
              'In a large bowl, whip heavy cream until stiff peaks form.',
              'Gently fold in sweetened condensed milk and vanilla extract.',
              'Pour mixture into a loaf pan or freezer-safe container.',
              'Freeze for at least 6 hours or until solid.',
              'Scoop and enjoy!'
            ],
            isVegetarian: true,
            isDrink: false,
          ),
          Recipe(
            title: 'Chocolate Milkshake',
            ingredients: [
              '2 cups vanilla ice cream',
              '1 cup milk',
              '1/4 cup chocolate syrup',
              'Whipped cream and chocolate shavings for topping (optional)'
            ],
            instructions: [
              'Combine ice cream, milk, and chocolate syrup in a blender.',
              'Blend until smooth and creamy.',
              'Pour into glasses and top with whipped cream and chocolate shavings, if desired.'
            ],
            isVegetarian: true,
            isDrink: true,
          ),
          Recipe(
            title: 'Strawberry Banana Smoothie',
            ingredients: [
              '1 cup frozen strawberries',
              '1 banana, sliced',
              '1 cup milk (dairy or non-dairy)',
              '1/2 cup yogurt (optional)',
              'Honey or maple syrup to taste (optional)'
            ],
            instructions: [
              'Combine all ingredients in a blender.',
              'Blend until smooth and creamy.',
              'Pour into glasses and enjoy!'
            ],
            isVegetarian: true,
            isDrink: true,
          ),
          Recipe(
            title: 'Affogato',
            ingredients: [
              '1 scoop vanilla ice cream',
              '1 shot of hot espresso'
            ],
            instructions: [
              'Place a scoop of vanilla ice cream in a small cup or glass.',
              'Pour a shot of hot espresso over the ice cream.',
              'Enjoy immediately while the ice cream is still melting.'
            ],
            isVegetarian: true,
            isDrink: true,
          ),
          Recipe(
            title: 'Pav Bhaji',
            ingredients: [
              'For Bhaji:',
              '3 tablespoons oil',
              '1 teaspoon mustard seeds',
              '1 teaspoon cumin seeds',
              '1/2 teaspoon turmeric powder',
              '1 tablespoon ginger-garlic paste',
              '1 onion, finely chopped',
              '2-3 tomatoes, finely chopped',
              '1 potato, boiled and mashed',
              '1 cup mixed vegetables (cauliflower, peas, carrots, beans)',
              '1 teaspoon pav bhaji masala',
              '1/2 teaspoon red chili powder',
              'Salt to taste',
              'For Pav:',
              'Pav (dinner rolls)',
              'Butter',
              'Chopped onions, cilantro, and lemon wedges for garnish'
            ],
            instructions: [
              'Heat oil in a pan. Add mustard seeds, cumin seeds, and turmeric powder. Let them splutter.',
              'Add ginger-garlic paste, onions, and saute until golden brown.',
              'Add tomatoes and cook until they become mushy.',
              'Add mashed potatoes, mixed vegetables, pav bhaji masala, chili powder, and salt. Cook for 5-7 minutes.',
              'Mash the bhaji well with a potato masher. Add water if needed to adjust the consistency.',
              'Toast pav on a pan with butter.',
              'Serve hot bhaji with buttered pav, garnished with chopped onions, cilantro, and lemon wedges.'
            ],
            isVegetarian: true,
            isDrink: false,
          ),
          Recipe(
            title: 'Pani Puri (Golgappa, Puchka)',
            ingredients: [
              'For Puris:',
              '1 cup semolina (sooji)',
              '1/4 cup whole wheat flour',
              'Salt to taste',
              'Oil for deep frying',
              'For Filling:',
              '1 cup boiled potatoes, mashed',
              '1/2 cup boiled chickpeas, mashed',
              '1/4 cup chopped onion',
              '1/4 cup chopped cilantro',
              '1/4 cup tamarind chutney',
              '1/4 cup green chutney',
              'For Pani:',
              '1 cup mint leaves',
              '1/2 cup cilantro',
              '1 green chili',
              '1 inch ginger',
              '1/4 teaspoon black salt',
              '1/4 teaspoon chaat masala',
              '1/4 teaspoon roasted cumin powder',
              'Salt to taste',
              'Water'
            ],
            instructions: [
              'For Puris:',
              'Knead a stiff dough with semolina, flour, salt, and water. Rest for 30 minutes.',
              'Divide into small balls, roll out thinly, and deep fry until puffed up and golden.',
              'For Filling:',
              'Mix mashed potatoes, chickpeas, onion, cilantro, and chutneys.',
              'For Pani:',
              'Blend all ingredients into a smooth paste. Add water to adjust consistency.',
              'To Assemble:',
              'Crack a small hole in the puri, fill with potato mixture, and dip in pani. Eat immediately.'
            ],
            isVegetarian: true,
            isDrink: false,
          ),
          Recipe(
            title: 'Samosa',
            ingredients: [
              'For the Dough:',
              '2 cups all-purpose flour',
              '1/4 cup oil or ghee',
              'Salt to taste',
              'Water (as needed)',
              'For the Filling:',
              '2-3 medium potatoes, boiled and mashed',
              '1/2 cup green peas',
              '1/4 cup chopped onions',
              '1/4 cup chopped cashews or peanuts',
              '1 green chili, finely chopped',
              '1 teaspoon ginger-garlic paste',
              '1/2 teaspoon cumin seeds',
              '1/2 teaspoon coriander powder',
              '1/2 teaspoon garam masala',
              '1/4 teaspoon amchur (dried mango powder)',
              '1/4 teaspoon red chili powder',
              'Salt to taste',
              'Oil for deep frying'
            ],
            instructions: [
              'For the Dough:',
              'Mix flour, oil/ghee, and salt. Add water gradually to form a stiff dough. Rest for 30 minutes.',
              'For the Filling:',
              'Heat oil in a pan. Add cumin seeds, chopped onions, and green chili. Sauté until golden.',
              'Add ginger-garlic paste, spices, mashed potatoes, and peas. Cook for 5-7 minutes.',
              'Divide the dough into small balls. Roll each ball into an oval shape.',
              'Cut the oval in half, form a cone, and fill with the potato mixture. Seal the edges.',
              'Deep fry the samosas until golden brown and crispy.'
            ],
            isVegetarian: true,
            isDrink: false,
          ),
          Recipe(
            title: 'Dabeli',
            ingredients: [
              'For the Dabeli Masala:',
              '1/4 cup roasted peanuts',
              '2 tablespoons dry coconut',
              '1 tablespoon sesame seeds',
              '1 teaspoon red chili powder',
              '1/2 teaspoon cumin powder',
              '1/2 teaspoon coriander powder',
              '1/4 teaspoon turmeric powder',
              '1/4 teaspoon garam masala',
              '1/4 teaspoon black salt',
              'A pinch of asafoetida (hing)',
              'Tamarind chutney',
              'Garlic chutney',
              'Green chutney',
              'Pav (dinner rolls)',
              'Sev (crispy chickpea flour noodles)',
              'Pomegranate seeds'
            ],
            instructions: [
              'Make the dabeli masala by grinding all the ingredients into a coarse powder.',
              'Boil and mash potatoes. Add dabeli masala, chutneys, and spices to the mashed potatoes. Mix well.',
              'Slit the pav horizontally. Spread the potato mixture on both sides.',
              'Roast on a griddle with butter until golden brown.',
              'Top with sev, pomegranate seeds, and chopped onions. Serve hot.'
            ],
            isVegetarian: true,
            isDrink: false,
          ),
          Recipe(
            title: 'Vada Pav',
            ingredients: [
              'For the Vada:',
              '2 cups boiled potatoes, mashed',
              '1/2 cup besan (gram flour)',
              '1 green chili, finely chopped',
              '1 teaspoon ginger-garlic paste',
              '1/2 teaspoon turmeric powder',
              '1/4 teaspoon red chili powder',
              'A pinch of asafoetida (hing)',
              'Salt to taste',
              'Oil for deep frying',
              'Pav (dinner rolls)',
              'Garlic chutney',
              'Green chutney',
              'Dry red chili powder (optional)'
            ],
            instructions: [
              'Mix mashed potatoes with besan, green chili, ginger-garlic paste, spices, and salt.',
              'Shape the mixture into small balls and deep fry until golden brown.',
              'Slit the pav horizontally. Spread chutneys on both sides.',
              'Place the fried vada inside the pav. Sprinkle dry red chili powder if desired.',
            ],
            isVegetarian: true,
            isDrink: false,
          ),
          Recipe(
            title: 'Burger',
            ingredients: [
              'Burger buns',
              'Ground beef or chicken (or vegetarian patty)',
              'Cheese slices',
              'Lettuce leaves',
              'Tomato slices',
              'Onion slices',
              'Pickles',
              'Ketchup',
              'Mustard'
            ],
            instructions: [
              'Cook the patty on a grill or pan until cooked through.',
              'Toast the burger buns lightly.',
              'Assemble the burger by placing the patty on the bottom bun, followed by cheese, lettuce, tomato, onion, and pickles.',
              'Add ketchup and mustard to taste.',
              'Top with the other bun and serve.'
            ],
            isVegetarian: false, // Can be vegetarian depending on the patty
            isDrink: false,
          ),
          Recipe(
            title: 'Veg Cheese Burger',
            ingredients: [
              'For the Patty:',
              '1 cup mashed potatoes',
              '1/2 cup grated carrots',
              '1/4 cup boiled green peas',
              '1/4 cup grated cheese (cheddar or mozzarella)',
              '2 tablespoons bread crumbs',
              '1 green chili, finely chopped (optional)',
              '1/2 teaspoon cumin powder',
              '1/2 teaspoon coriander powder',
              '1/4 teaspoon turmeric powder',
              '1/4 teaspoon red chili powder',
              'Salt to taste',
              'Oil for shallow frying',
              'Burger buns',
              'Cheese slices',
              'Lettuce leaves',
              'Tomato slices',
              'Onion slices',
              'Pickles',
              'Ketchup',
              'Mayonnaise'
            ],
            instructions: [
              'For the Patty:',
              'Combine mashed potatoes, grated carrots, green peas, grated cheese, bread crumbs, spices, and salt.',
              'Mix well and form into patties of desired size and thickness.',
              'Heat oil in a pan and shallow fry the patties until golden brown on both sides.',
              'To Assemble:',
              'Lightly toast the burger buns.',
              'Spread mayonnaise and ketchup on the bottom bun.',
              'Place a lettuce leaf, a tomato slice, an onion slice, and a pickle on the bottom bun.',
              'Add the cooked patty and a slice of cheese on top.',
              'Place the top bun and serve hot.'
            ],
            isVegetarian: true,
            isDrink: false,
          ),
          Recipe(
            title: 'Barfi (Milk Fudge)',
            ingredients: [
              '1 cup milk powder (khoya)',
              '1/2 cup sugar',
              '1/4 cup ghee (clarified butter)',
              '1/4 cup milk',
              'Chopped nuts (almonds, pistachios) for garnish (optional)'
            ],
            instructions: [
              'Grease a tray or plate with ghee.',
              'Heat ghee in a pan over low heat.',
              'Add milk powder and sugar. Stir continuously until the mixture thickens and leaves the sides of the pan.',
              'Add milk and continue stirring until the mixture becomes smooth and forms a soft ball.',
              'Pour the mixture onto the greased tray and spread evenly.',
              'Garnish with chopped nuts (optional).',
              'Let it cool and set completely.',
              'Cut into desired shapes and serve.'
            ],
            isVegetarian: true,
            isDrink: false,
          ),
          Recipe(
            title: 'Cham Cham',
            ingredients: [
              'For Chhana (Cheese):',
              '1 liter whole milk',
              '1/4 cup lemon juice or vinegar',
              'For Sugar Syrup:',
              '2 cups sugar',
              '3 cups water',
              'A few strands of saffron',
              '2-3 cardamom pods',
              'Yellow food color (optional)',
              'Desiccated coconut for coating'
            ],
            instructions: [
              'To make chhana, boil milk, add lemon juice, and stir until milk curdles. Strain through a cheesecloth.',
              'Knead the chhana for a few minutes until smooth. Divide and shape into small cylindrical pieces.',
              'Cook the chhana pieces in boiling water for 10-15 minutes until they double in size.',
              'Make sugar syrup by boiling sugar, water, saffron, and cardamom until it thickens slightly. Add food color (optional).',
              'Soak the cooked chhana pieces in the sugar syrup for at least 2 hours.',
              'Remove from syrup, squeeze gently, and roll in desiccated coconut. Serve chilled.'
            ],
            isVegetarian: true,
            isDrink: false,
          ),
          Recipe(
            title: 'Kaju Katli (Cashew Fudge)',
            ingredients: [
              '1 cup cashew nuts',
              '1/2 cup sugar',
              '1/4 cup water',
              '1/4 teaspoon cardamom powder',
              'Silver leaf (varq) for garnish (optional)'
            ],
            instructions: [
              'Grind cashew nuts into a fine powder.',
              'Prepare sugar syrup by boiling sugar and water until it reaches a single thread consistency.',
              'Add cashew powder and cardamom powder to the syrup. Stir continuously until the mixture thickens and leaves the sides of the pan.',
              'Pour the mixture onto a greased plate or tray and spread evenly.',
              'When slightly cooled, cut into diamond shapes.',
              'Garnish with silver leaf (optional) and let it cool completely before serving.'
            ],
            isVegetarian: true,
            isDrink: false,
          ),
          Recipe(
            title: 'Soan Papdi',
            ingredients: [
              '1 cup gram flour (besan)',
              '1 cup ghee (clarified butter)',
              '1 cup sugar',
              '1/2 cup water',
              '1/4 teaspoon cardamom powder',
              'A few strands of saffron (optional)'
            ],
            instructions: [
              'Roast gram flour in ghee until it turns golden brown.',
              'Prepare sugar syrup by boiling sugar and water until it reaches a one-thread consistency.',
              'Add cardamom powder and saffron (optional) to the syrup.',
              'Pour the sugar syrup into the roasted gram flour mixture and mix well.',
              'Transfer the mixture to a greased plate and let it cool slightly.',
              'Knead the mixture until it becomes fluffy and flaky.',
              'Cut into desired shapes and serve.'
            ],
            isVegetarian: true,
            isDrink: false,
          ),
          Recipe(
            title: 'Veggie Puff',
            ingredients: [
              'Puff pastry sheets',
              'For the Filling:',
              '1 cup mixed vegetables (carrots, peas, corn, beans), finely chopped',
              '1/2 onion, finely chopped',
              '1 green chili, finely chopped (optional)',
              '1/2 teaspoon ginger-garlic paste',
              '1/2 teaspoon cumin powder',
              '1/2 teaspoon coriander powder',
              '1/4 teaspoon turmeric powder',
              '1/4 teaspoon red chili powder (optional)',
              'Salt to taste',
              'Oil for frying',
              '(Optional toppings: grated cheese, ketchup, mint-coriander chutney)'
            ],
            instructions: [
              'Cook the filling by heating oil in a pan, adding spices, and then vegetables.',
              'Cook until vegetables are softened and the mixture thickens.',
              'Cut puff pastry sheets into squares.',
              'Place a spoonful of filling in the center of each square.',
              'Fold the pastry diagonally to form a triangle, sealing the edges.',
              'Deep-fry the puffs until golden brown and puffed up.',
              'Serve hot with optional toppings.'
            ],
            isVegetarian: true,
            isDrink: false,
          ),
          Recipe(
            title: 'Veggie Hot Dog',
            ingredients: [
              'Hot dog buns',
              'Vegetarian hot dogs (soy or tofu-based)',
              '(Optional toppings: grated cheese, ketchup, mustard, mayonnaise, relish)'
            ],
            instructions: [
              'Cook the vegetarian hot dogs according to package instructions (grill, pan-fry, or boil).',
              'Warm the hot dog buns.',
              'Place the hot dog in the bun.',
              'Top with desired toppings.'
            ],
            isVegetarian: true,
            isDrink: false,
          ),
          Recipe(
            title: 'Veggie Frankie (Wrap)',
            ingredients: [
              'Whole wheat tortillas or chapatis',
              'For the Filling:',
              'Mixed vegetables (potatoes, cauliflower, peas), cooked and mashed',
              '1/2 onion, finely chopped',
              '1 green chili, finely chopped (optional)',
              '1/2 teaspoon ginger-garlic paste',
              '1/2 teaspoon cumin powder',
              '1/2 teaspoon coriander powder',
              '1/4 teaspoon turmeric powder',
              '1/4 teaspoon red chili powder (optional)',
              'Salt to taste',
              'Oil for cooking',
              '(Optional toppings: grated cheese, mayonnaise, ketchup, mint-coriander chutney)'
            ],
            instructions: [
              'Cook the filling by heating oil in a pan, adding spices, and then vegetables. Cook until the mixture thickens.',
              'Spread the filling evenly on a tortilla or chapati.',
              'Add optional toppings.',
              'Roll the tortilla tightly to form a Frankie.',
              'Grill or toast the Frankie until heated through and slightly crispy.',
            ],
            isVegetarian: true,
            isDrink: false,
          ),
          Recipe(
            title: 'Veggie Sandwich',
            ingredients: [
              'Bread slices',
              'Butter or mayonnaise',
              'Sliced cucumber',
              'Sliced tomatoes',
              'Onion slices',
              'Grated cheese (optional)',
              'Lettuce leaves',
              'Salt and pepper to taste',
              '(Optional additions: avocado slices, sprouts, boiled potatoes, beetroot slices)'
            ],
            instructions: [
              'Spread butter or mayonnaise on both bread slices.',
              'Layer the vegetables and cheese (optional) on one slice.',
              'Season with salt and pepper.',
              'Cover with the other bread slice and cut into halves.'
            ],
            isVegetarian: true,
            isDrink: false,
          ),
          Recipe(
            title: 'Sabudana Khichdi',
            ingredients: [
              '1 cup sabudana (tapioca pearls)',
              '1/2 cup roasted peanuts',
              '2 medium potatoes, boiled and cubed',
              '1 green chili, finely chopped',
              '1/4 cup chopped cilantro',
              '1 teaspoon cumin seeds',
              '1/2 teaspoon turmeric powder',
              'Salt to taste',
              'Lemon juice (optional)',
              '2 tablespoons ghee or oil'
            ],
            instructions: [
              'Soak sabudana for 4-5 hours until they turn soft and fluffy.',
              'Heat ghee in a pan. Add cumin seeds, green chili, and peanuts. Saute until fragrant.',
              'Add potatoes and turmeric powder. Cook for a few minutes.',
              'Add soaked sabudana and salt. Mix well and cook until the pearls turn translucent.',
              'Garnish with cilantro and drizzle lemon juice (optional) before serving.'
            ],
            isVegetarian: true,
            isDrink: false,
          ),
          Recipe(
            title: 'Farali Pattice',
            ingredients: [
              '1 cup boiled and mashed potatoes',
              '1/2 cup roasted peanuts, coarsely ground',
              '1 green chili, finely chopped',
              '1/4 cup chopped cilantro',
              '1 teaspoon cumin powder',
              '1/2 teaspoon red chili powder',
              '1/4 teaspoon amchur (dried mango powder)',
              'Salt to taste',
              'Oil for shallow frying'
            ],
            instructions: [
              'Combine all ingredients in a bowl and mix well.',
              'Shape the mixture into small patties.',
              'Heat oil in a pan and shallow fry the patties until golden brown on both sides.',
              'Serve hot with green chutney or yogurt.'
            ],
            isVegetarian: true,
            isDrink: false,
          ),
          Recipe(
            title: 'Rajgira Sheera',
            ingredients: [
              '1/2 cup rajgira flour (amaranth flour)',
              '1/4 cup ghee',
              '1/2 cup sugar',
              '1 cup milk',
              '1/4 cup chopped nuts (cashews, almonds)',
              '1/4 teaspoon cardamom powder'
            ],
            instructions: [
              'Heat ghee in a pan. Add rajgira flour and roast until fragrant.',
              'Add milk and sugar. Stir continuously until the mixture thickens.',
              'Add chopped nuts and cardamom powder. Mix well.',
              'Serve warm as a dessert.'
            ],
            isVegetarian: true,
            isDrink: false,
          ),
          Recipe(
            title: 'Mitha Samo/Moraiya',
            ingredients: [
              '1 cup samo (barnyard millet)',
              '2 cups milk',
              '1/4 cup sugar (or jaggery for a healthier option)',
              '1/4 cup chopped nuts (cashews, almonds, raisins)',
              '1/4 teaspoon cardamom powder',
              'A pinch of saffron strands (optional)',
              '1 tablespoon ghee'
            ],
            instructions: [
              'Rinse samo thoroughly and soak in water for 30 minutes.',
              'Heat ghee in a pan. Add drained samo and roast for a few minutes.',
              'Add milk, sugar (or jaggery), and nuts. Bring to a boil.',
              'Reduce heat and simmer until the milk is absorbed and samo is cooked.',
              'Add cardamom powder and saffron (optional). Mix well.',
              'Serve warm as a dessert.'
            ],
            isVegetarian: true,
            isDrink: false,
          ),
          Recipe(
            title: 'Khara Samo/Moraiya',
            ingredients: [
              '1 cup samo (barnyard millet)',
              '2 cups water',
              '1 tablespoon ghee or oil',
              '1 teaspoon cumin seeds',
              '1 green chili, finely chopped',
              '1/2 inch ginger, grated',
              '1/4 cup roasted peanuts',
              '1/4 cup chopped coriander leaves',
              'Salt to taste'
            ],
            instructions: [
              'Rinse samo thoroughly and soak in water for 30 minutes.',
              'Heat ghee in a pan. Add cumin seeds, green chili, and ginger. Sauté until fragrant.',
              'Add drained samo and peanuts. Roast for a few minutes.',
              'Add 2 cups of water and salt. Bring to a boil, then reduce heat and simmer until the water is absorbed and samo is cooked.',
              'Garnish with coriander leaves and serve hot.'
            ],
            isVegetarian: true,
            isDrink: false,
          ),
          Recipe(
            title: 'Chapati/Roti',
            ingredients: [
              '2 cups whole wheat flour (atta)',
              'Water (as needed for kneading)',
              'Salt (optional)',
              'Ghee or oil (for cooking)'
            ],
            instructions: [
              'In a bowl, mix whole wheat flour and salt (if using).',
              'Gradually add water and knead into a soft dough. Rest for 15-20 minutes.',
              'Divide the dough into small balls.',
              'Roll out each ball into a thin circle on a floured surface.',
              'Heat a flat pan or tava. Cook each chapati until small bubbles appear.',
              'Flip and cook the other side until brown spots appear.',
              'Optionally, apply ghee or oil on both sides while cooking.'
            ],
            isVegetarian: true,
            isDrink: false,
          ),
          Recipe(
            title: 'Rotla/Rolto',
            ingredients: [
              '1 cup bajra flour (pearl millet flour)',
              '1/2 cup wheat flour (optional, for better binding)',
              'Salt to taste',
              'Warm water (as needed for kneading)',
              'Oil or ghee (for cooking)'
            ],
            instructions: [
              'Mix bajra flour, wheat flour (if using), and salt in a bowl.',
              'Gradually add warm water and knead into a semi-soft dough.',
              'Divide the dough into small balls.',
              'Roll out each ball into a thick circle using your hands or a rolling pin.',
              'Heat a tava or flat pan. Cook the rotla until cooked through and brown spots appear on both sides.',
              'Serve hot with ghee or oil.'
            ],
            isVegetarian: true,
            isDrink: false,
          ),
          Recipe(
            title: 'Puri',
            ingredients: [
              '2 cups whole wheat flour (atta)',
              'Water (as needed for kneading)',
              'Salt to taste',
              'Oil for deep frying'
            ],
            instructions: [
              'In a bowl, mix whole wheat flour and salt.',
              'Gradually add water and knead into a stiff dough. Rest for 15-20 minutes.',
              'Divide the dough into small balls.',
              'Roll out each ball into a thin circle on a floured surface.',
              'Heat oil in a deep pan.',
              'Fry each puri until it puffs up and turns golden brown on both sides.'
            ],
            isVegetarian: true,
            isDrink: false,
          ),
          Recipe(
            title: 'Thepla',
            ingredients: [
              '2 cups whole wheat flour (atta)',
              '1/2 cup chopped fenugreek leaves (methi)',
              '1/4 cup grated jaggery',
              '1 tablespoon sesame seeds',
              '1/2 teaspoon turmeric powder',
              '1/2 teaspoon red chili powder',
              'Salt to taste',
              'Oil or ghee (for cooking)'
            ],
            instructions: [
              'In a bowl, mix whole wheat flour, fenugreek leaves, jaggery, sesame seeds, turmeric powder, chili powder, and salt.',
              'Gradually add water and knead into a soft dough. Rest for 15-20 minutes.',
              'Divide the dough into small balls.',
              'Roll out each ball into a thin circle on a floured surface.',
              'Heat a tava or flat pan. Cook each thepla until brown spots appear on both sides.',
              'Apply oil or ghee on both sides while cooking.'
            ],
            isVegetarian: true,
            isDrink: false,
          ),
          Recipe(
            title: 'Moong Dal Khichdi',
            ingredients: [
              '1/2 cup rice',
              '1/4 cup split green moong dal (yellow moong dal can also be used)',
              '1 teaspoon ghee or oil',
              '1/2 teaspoon cumin seeds',
              'A pinch of asafoetida (hing, optional)',
              '1/2 teaspoon turmeric powder',
              '1 green chili, finely chopped (optional)',
              '1-inch ginger, grated (optional)',
              '2 cups water',
              'Salt to taste',
              'Fresh coriander leaves for garnish'
            ],
            instructions: [
              'Wash and soak rice and moong dal together for 30 minutes.',
              'Heat ghee in a pressure cooker. Add cumin seeds, hing (if using), and let them splutter.',
              'Add green chili and ginger (if using) and sauté for a few seconds.',
              'Add turmeric powder and soaked rice and moong dal. Mix well.',
              'Add water and salt. Close the pressure cooker lid and cook for 3-4 whistles.',
              'Let the pressure release naturally. Open the lid and fluff the khichdi with a fork.',
              'Garnish with coriander leaves and serve hot with yogurt or pickle.'
            ],
            isVegetarian: true,
            isDrink: false,
          ),
          Recipe(
            title: 'Poha/Pauva',
            ingredients: [
              '2 cups thick poha (flattened rice)',
              '2 tablespoons oil or ghee',
              '1 teaspoon mustard seeds',
              '1 teaspoon cumin seeds',
              '1/2 teaspoon turmeric powder',
              '1 green chili, finely chopped (optional)',
              '1/4 cup chopped onion (optional)',
              '2-3 curry leaves',
              '1/4 cup chopped peanuts',
              'Salt to taste',
              'Lemon juice to taste',
              'Chopped coriander leaves for garnish'
            ],
            instructions: [
              'Rinse poha under running water until soft but not mushy. Drain well.',
              'Heat oil in a pan. Add mustard seeds and cumin seeds. Let them splutter.',
              'Add green chili (if using), onion (if using), and curry leaves. Saute for a minute.',
              'Add turmeric powder and peanuts. Stir briefly.',
              'Add the soaked poha and salt. Mix gently to avoid breaking the poha.',
              'Cook for 2-3 minutes until heated through.',
              'Sprinkle with lemon juice and garnish with coriander leaves.',
              'Serve hot.'
            ],
            isVegetarian: true,
            isDrink: false,
          ),
          Recipe(
            title: 'Nimbu Pani (Lemon Water)',
            ingredients: [
              'Juice of 2-3 lemons',
              '4 cups water',
              'Sugar to taste',
              'A pinch of black salt (kala namak)',
              'A few mint leaves (optional)',
              'Ice cubes'
            ],
            instructions: [
              'Combine lemon juice, water, sugar, and black salt in a pitcher.',
              'Stir until the sugar is dissolved.',
              'Add mint leaves (optional) and ice cubes.',
              'Serve chilled.'
            ],
            isVegetarian: true,
            isDrink: true,
          )
        ],
      ),
    );
  }
}
