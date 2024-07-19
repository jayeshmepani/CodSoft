import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../providers/task_provider.dart';
import '../models/task.dart';

class AddTaskScreen extends StatefulWidget {
  final Task? task;

  AddTaskScreen({this.task});

  @override
  _AddTaskScreenState createState() => _AddTaskScreenState();
}

class _AddTaskScreenState extends State<AddTaskScreen> {
  final _controller = TextEditingController();

  @override
  void initState() {
    if (widget.task != null) {
      _controller.text = widget.task!.title;
    }
    super.initState();
  }

  @override
  Widget build(BuildContext context) {
    final taskProvider = Provider.of<TaskProvider>(context);

    return Scaffold(
      appBar: AppBar(
        title: Text(widget.task == null ? 'Add Task' : 'Edit Task'),
      ),
      body: Padding(
        padding: EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            TextField(
              controller: _controller,
              decoration: InputDecoration(labelText: 'Task Title'),
            ),
            SizedBox(height: 20),
            ElevatedButton(
              onPressed: () {
                final newTitle = _controller.text.trim();
                if (newTitle.isEmpty) {
                  ScaffoldMessenger.of(context).showSnackBar(
                    SnackBar(content: Text('Please enter a task title')),
                  );
                  return;
                }

                if (widget.task == null) {
                  taskProvider.addTask(Task(id: '', title: newTitle)); // Pass an empty id or generate one
                } else {
                  taskProvider.editTask(widget.task!, newTitle);
                }
                Navigator.of(context).pop();
              },
              child: Text(widget.task == null ? 'Add Task' : 'Save Changes'),
            ),
            SizedBox(height: 10), // Add space between buttons
            if (widget.task != null)
              ElevatedButton(
                onPressed: () {
                  taskProvider.deleteTask(widget.task!);
                  Navigator.of(context).pop();
                },
                style: ElevatedButton.styleFrom(
                  backgroundColor: Colors.red, // Change the color to red for the delete button
                ),
                child: Text('Delete Task'),
              ),
          ],
        ),
      ),
    );
  }
}
