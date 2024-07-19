import 'package:flutter/material.dart';
import '../models/budget.dart';

class BudgetProvider with ChangeNotifier {
  Budget _budget = Budget(amount: 0.0);

  Budget get budget => _budget;

  void setBudget(double amount) {
    _budget = Budget(amount: amount);
    notifyListeners();
  }
}
