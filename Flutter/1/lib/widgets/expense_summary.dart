import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../providers/transactions.dart';

class ExpenseSummary extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    final transactions = Provider.of<Transactions>(context);

    return Container(
      padding: EdgeInsets.all(16.0),
      child: Column(
        children: [
          Text(
            'Total Expenses: \$${transactions.totalExpenses.toStringAsFixed(2)}',
            style: TextStyle(fontSize: 20),
          ),
        ],
      ),
    );
  }
}
