from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [("accounts", "0001_initial")]  # noqa: RUF012
    operations = [  # noqa: RUF012
        migrations.CreateModel(
            name="Post",
            fields=[
                ("id", models.AutoField(primary_key=True)),
                ("title", models.CharField(max_length=200)),
                ("author", models.ForeignKey("accounts.User", on_delete=models.CASCADE)),
            ],
        ),
    ]
