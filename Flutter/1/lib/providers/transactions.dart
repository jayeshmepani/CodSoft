import 'package:flutter/foundation.dart';
import '../models/transaction.dart';

class Transactions with ChangeNotifier {
  List<Transaction> _transactions = [];

  List<Transaction> get transactions {
    return [..._transactions];
  }

  void addTransaction(Transaction transaction) {
    _transactions.add(transaction);
    notifyListeners();
  }

  void removeTransaction(String id) {
    _transactions.removeWhere((tx) => tx.id == id);
    notifyListeners();
  }

  double get totalExpenses {
    return _transactions.fold(0.0, (sum, tx) => sum + tx.amount);
  }
}
