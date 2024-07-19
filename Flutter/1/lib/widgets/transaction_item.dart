import 'package:flutter/material.dart';
import '../models/transaction.dart';

class TransactionItem extends StatelessWidget {
  final Transaction transaction;

  TransactionItem(this.transaction);

  @override
  Widget build(BuildContext context) {
    return ListTile(
      title: Text(transaction.title),
      subtitle: Text(transaction.category),
      trailing: Text('- \$${transaction.amount.toStringAsFixed(2)}'),
    );
  }
}
