from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies: list = []  # noqa: RUF012
    operations = [  # noqa: RUF012
        migrations.CreateModel(
            name="User",
            fields=[
                ("id", models.AutoField(primary_key=True)),
                ("email", models.CharField(max_length=255)),
            ],
        ),
    ]
