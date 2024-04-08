:mod:`{{module}}`.{{objname}}
{{ underline }}==============

.. currentmodule:: {{module}}

.. autoclass:: {{ objname }}

   {% block methods %}
    .. automethod:: __call__
    .. automethod:: load_default_package
    .. automethod:: load_model
   {% endblock %}

.. _sphx_glr_backref_{{module}}.{{objname}}:

.. minigallery:: {{module}}.{{objname}}
      :add-heading:
      :heading-level: ^

.. raw:: html

    <div class="clearer"></div>