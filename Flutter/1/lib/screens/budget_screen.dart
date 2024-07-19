import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../providers/budget_provider.dart';

class BudgetScreen extends StatefulWidget {
  @override
  _BudgetScreenState createState() => _BudgetScreenState();
}

class _BudgetScreenState extends State<BudgetScreen> {
  final TextEditingController _controller = TextEditingController();

  @override
  Widget build(BuildContext context) {
    final budgetProvider = Provider.of<BudgetProvider>(context);

    return Scaffold(
      appBar: AppBar(title: Text('Set Monthly Budget')),
      body: Padding(
        padding: EdgeInsets.all(16.0),
        child: Column(
          children: [
            Text('Current Budget: \$${budgetProvider.budget.amount.toStringAsFixed(2)}'),
            TextField(
              controller: _controller,
              decoration: InputDecoration(labelText: 'Enter Budget Amount'),
              keyboardType: TextInputType.number,
            ),
            SizedBox(height: 20),
            ElevatedButton(
              onPressed: () {
                final amount = double.parse(_controller.text);
                budgetProvider.setBudget(amount);
                Navigator.pop(context);
              },
              child: Text('Set Budget'),
            ),
          ],
        ),
      ),
    );
  }
}
