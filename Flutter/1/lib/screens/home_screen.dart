import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../providers/budget_provider.dart';
import '../widgets/expense_summary.dart';
import '../widgets/transaction_list.dart';
import 'add_transaction_screen.dart';
import 'budget_screen.dart';

class HomeScreen extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    final budgetProvider = Provider.of<BudgetProvider>(context);

    return Scaffold(
      appBar: AppBar(title: Text('Personal Expense Tracker')),
      body: Column(
        children: [
          Center(
            child: Column(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                Text('Monthly Budget: \$${budgetProvider.budget.amount.toStringAsFixed(2)}'),
                SizedBox(height: 20),
                ElevatedButton(
                  onPressed: () {
                    Navigator.push(
                      context,
                      MaterialPageRoute(builder: (context) => BudgetScreen()),
                    );
                  },
                  child: Text('Set Budget'),
                ),
              ],
            ),
          ),
          ExpenseSummary(),
          TransactionList(),
        ],
      ),
      floatingActionButton: FloatingActionButton(
        onPressed: () {
          Navigator.of(context).push(
            MaterialPageRoute(
              builder: (context) => AddTransactionScreen(),
            ),
          );
        },
        child: Icon(Icons.add),
      ),
    );
  }
}
