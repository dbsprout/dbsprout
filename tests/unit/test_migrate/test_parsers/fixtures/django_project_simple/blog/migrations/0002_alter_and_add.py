from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [("blog", "0001_initial")]  # noqa: RUF012
    operations = [  # noqa: RUF012
        migrations.AddField(
            model_name="Post",
            name="subtitle",
            field=models.CharField(max_length=100, null=True),
        ),
        migrations.AlterField(
            model_name="Post",
            name="title",
            field=models.CharField(max_length=300),
        ),
    ]
